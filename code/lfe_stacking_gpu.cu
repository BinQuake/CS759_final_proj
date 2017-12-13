/*
LFE stacking with GPU

Implemented two version of stacking same as CPU version

1. PWS stacking
	Required package, cufft to do fft analysis find the phase information of each sac trace
2. Leaner stacking
	Simply stacking all traces with same starting point

Input arguments
	1. sac_file_list
	2. output name for pws stacking
	3. output name for linear stacking

Reference:
 	sacio: developed by Dr. Lupei Zhu http://www.eas.slu.edu/People/LZhu/home.htm
	pws to LFE: Thurber et al., 2014, Phase-weighted stacking applied to low-frequency earthquakes. Bull. Seismol. Soc. Am., 104(5), 2567-2572
  	tf-pws: Schimmel M. and J. Gallart, 2007, Frequency-dependent phase coherence for noise suppression in seismic array data. J. Geophys. Res. 112, B04303
  	fast s transform: Brown R. et al., 2010, A general description of linear time-frequency transforms and formulation of a fast, invertible transform that samples the continuous S-transform spectrum nonredundantly. IEEE Transactions on Signal Processing, 58, 281-290.

PWS stacking part code referenced from Xiangfang Zeng's original PWS stacking code.
Initial PWS stacking code has a problem with arbitary array length. Even though it did padding for array to enlarge to next power of 2. The output is not correct.
Need to preprocess the sac traces with power of 2 number of points.
*/
#include <cufft.h>
#include <time.h>
#include "sacio.c"
#define NMAX 1024
#define MAX 8388608
#define MAX_TH 1024
#define MAX_BK 65536 
//kernel functions

__device__ inline void atomicFloatAdd(float* address, float value)
// define atomicAdd for float operations
{

  float old = value;  
  float new_old;
  do
  {
	new_old = atomicExch(address, 0.0f);
	new_old += old;
  }
  while ((old = atomicExch(address, new_old))!=0.0f);
};
__global__ void linear_stack(float *in, float *out, int npts){
	// for the input matrix in, the max number threads is 1024 for each block
	// if the npts is larger than 1024 then need to launch more block for this
	// then, the blockIdx.y is used to work with the thread number
	// blockId.x is used to work with the number of traces
	int indx = blockIdx.y*blockDim.x + threadIdx.x;
	int trace_id = blockIdx.x;
	
	if (indx < npts) {
		atomicFloatAdd(out+indx, in[trace_id*npts+indx]);
	}
}

__global__ void sum_along_col(cufftComplex *sum,cufftComplex *s,int n)
{
	int i;
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	while(idx<n)
	{
		sum[idx].x=0.0;
		sum[idx].y=0.0;
		for(i=0;i<n;i++)
		{
			sum[idx].x = sum[idx].x + s[idx*n+i].x;
			sum[idx].y = sum[idx].y + s[idx*n+i].y;
		}
		if(cuCabsf(sum[idx])){sum[idx].x=0.0;sum[idx].y=0.0;}
	}
}
__global__ void copy_complex_vector(cufftComplex *a,cufftComplex *b,int n)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	while(idx<n)
	{
                a[idx].x=b[idx].x;
                a[idx].y=b[idx].y;
                idx=idx+blockDim.x*gridDim.x;
	}
}
__global__ void copy_complex_vector_shift(cufftComplex *a,cufftComplex *b,int n,int shift)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	int idy=threadIdx.x+blockIdx.x*blockDim.x+shift;
	if(idy>=n) idy=idy-n;
	while(idx<n)
	{
		
                a[idx].x=b[idy].x;
                a[idx].y=b[idy].y;
                idx=idx+blockDim.x*gridDim.x;
	}
}
__global__ void copy_complex_matrix_shift(cufftComplex *a,cufftComplex *b,int n)
{
	//b is a new_npts long vector
	//a is a new_npt*new_npts matrix
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;
	int idx,idb;
	idx=x+y*n;
	idb=x+y;
	if(idb>=n) idb=idb-n;
                a[idx].x=b[idb].x;
                a[idx].y=b[idb].y;
}
__global__ void copy_complex_matrix(cufftComplex *a,cufftComplex *b,int n)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	while(idx<n*n)
	{
                a[idx].x=b[idx].x;
                a[idx].y=b[idx].y;
                idx=idx+blockDim.x*gridDim.x;
	}
}
__global__ void complex_mul_float_mat(cufftComplex *a,float *b,int n)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	while(idx<n*n)
	{
                a[idx].x=a[idx].x*b[idx];
                a[idx].y=a[idx].y*b[idx];
                idx=idx+blockDim.x*gridDim.x;
	}
}
__global__ void update_smat(cufftComplex *s,cufftComplex *ft,int n)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	float abs;
	float tmpx,tmpy;
	while(idx<n*n)
	{
		abs=cuCabsf(s[idx]);
		if(isnan(abs)) abs=1;
                tmpx=(s[idx].x * ft[idx].x - s[idx].y*ft[idx].y)/abs;
                tmpy=(s[idx].y * ft[idx].x + s[idx].x*ft[idx].y)/abs;
                s[idx].x=tmpx;
		s[idx].y=tmpy;
                idx=idx+blockDim.x*gridDim.x;

	}
}
__global__ void sum_smat(cufftComplex *sum,cufftComplex *s,int n)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	while(idx<n*n)
	{
                sum[idx].x=sum[idx].x+s[idx].x;
                sum[idx].y=sum[idx].y+s[idx].y;
                idx=idx+blockDim.x*gridDim.x;
	}
}
//wght is a complex whereas real weight is wght.x
__global__ void mul_wght_mat(cufftComplex *s,cufftComplex *w,int n)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	while(idx<n*n)
	{
                s[idx].x=s[idx].x*w[idx].x;
                s[idx].y=s[idx].y*w[idx].x;
                idx=idx+blockDim.x*gridDim.x;
	}
}
__global__ void pws_wght_mat(cufftComplex *s,int n,int ntr,int pwr)
{
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	float abs;
	while(idx<n*n)
	{
		abs=sqrt(s[idx].x*s[idx].x+s[idx].y*s[idx].y);
		s[idx].x=abs*abs/ntr; //when pwr = 2
                idx=idx+blockDim.x*gridDim.x;
	}	
}
////main function and other cpu subroutines 
int main(int ac,char **argv)
{
	int nextpow2(int);
	void read_trace_list(char*,float**,int*,int*,float*,float*);

	//n*n matrix operators
	void gaussian_matrix(float*,int,float*,int);
	void pws_wght(cufftComplex*,int,int,int);
	void complex_mul_float(cufftComplex *,float*,int);
	void update_s_mat(cufftComplex*,cufftComplex*,int);
	void sum_s_mat(cufftComplex*,cufftComplex*,int);
	void mul_wght(cufftComplex*,cufftComplex*,int);

	
	if(ac!=4)
	{
		fprintf(stderr,"xxx input_list pws_sac linear_sac\n");
		exit(-1);
	}

	//timers
        clock_t t1,t2,t3,t5,t7;
	double time_cost;       

	int k=2;
	int pwr=2;
	//input data
	int ntrace;
	int new_npts,npts;
	float dt,b;
	float **dat;
	dat=(float**)malloc(sizeof(float*)*NMAX);
	
	int i,j;
	//cufft
	cufftHandle plan0,plan1,plan2; //plan0: 1 traces, plan1: all input traces batch=nstrace, plan2: each trace tf ifft batch=new_npts/nloop
	//storage spectra of all traces
	cufftComplex *host_dat;
	//store spectrum of linear-stacking traces
	cufftComplex *host_stack;
	cufftComplex *dev_stack;
	//temp variable of spectrum
//	cufftComplex *host_tmp;
	cufftComplex *dev_tmp;

	cufftComplex *dev_dat;
	cufftResult res;
	//control parameters of s trans. 
	int batch,nloop,shift;
	//for matrix new_npts*new_npts
	int threads,blocks;

	//gaussian matrix
	float df;
	float *freq_vec;
	float *time_vec;
	float *w_mat;
	float *dev_w_mat;
	
	//s matrix
	cufftComplex *s_mat,*dev_s_mat,*dev_sum_s;
	//ft matrix
	cufftComplex *ft_mat,*dev_ft_mat;
	float ft_tmp;
	float pi=3.14159;

	//read input data	
	t1=clock();
	read_trace_list(argv[1],dat,&ntrace,&npts,&dt,&b);
	fprintf(stderr,"trace no: %d npts: %d dt: %f\n",ntrace,npts,dt);
	t2=clock();
	new_npts=nextpow2(npts);

	//we know threads and blocks
	if(new_npts >= MAX_TH)
		threads=MAX_TH;
	else
		threads=new_npts;
	blocks=new_npts*new_npts/threads;
	if(blocks > MAX_BK) blocks=MAX_BK;
	//set up bk and th in copy_complex_matrix_shift
	int nth=32;
	int nbk=new_npts/nth;
	dim3 th(nth,nth);
	dim3 bk(nbk,nbk);	
	
	t3 = clock();
	df=1/((new_npts-1)*dt);
	freq_vec=(float*)malloc(sizeof(float)*new_npts);
	time_vec=(float*)malloc(sizeof(float)*new_npts);
	ft_mat=(cufftComplex*)malloc(sizeof(cufftComplex)*new_npts*new_npts);
	cudaMalloc((void**)&dev_ft_mat,sizeof(float)*new_npts*new_npts);
	for(i=0;i<new_npts/2;i++)
	{	
		freq_vec[i]=i*df;
		time_vec[i]=i*dt;
	}
	for(;i<new_npts;i++)
	{
		freq_vec[i]=(new_npts-i)*df;
		time_vec[i]=i*dt;
	}
	for(i=0;i<new_npts;i++)
	{
		for(j=0;j<new_npts;j++)
		{

			ft_tmp = freq_vec[j]*time_vec[i];
			ft_mat[j*new_npts+i].y=sin(ft_tmp*2*pi);
			ft_mat[j*new_npts+i].x=cos(ft_tmp*2*pi);
		}
	}
//	fprintf(stderr,"set up ft matrix\n");
	w_mat=(float*)malloc(sizeof(float)*new_npts*new_npts);
	cudaMalloc((void**)&dev_w_mat,sizeof(float)*new_npts*new_npts);
	
	//cufft of input data
	cudaMalloc((void**)&dev_dat,sizeof(cufftComplex)*ntrace*new_npts);
	cudaMalloc((void**)&dev_stack,sizeof(cufftComplex)*new_npts);
	int n1[1]={new_npts};

	//pinned 
	cudaHostAlloc((void**)&host_dat,sizeof(cufftComplex)*ntrace*new_npts,cudaHostAllocDefault);

	cudaMalloc((void**)&dev_tmp,sizeof(float)*2*new_npts);
	//pinned 
	cudaHostAlloc((void**)&host_stack,sizeof(cufftComplex)*ntrace*new_npts,cudaHostAllocDefault);
	for(i=0;i<ntrace;i++)
	{
		for(j=0;j<npts;j++)
		{
		if(i!=0)
			host_stack[j].x=host_stack[j].x+dat[i][j];
		else
			host_stack[j].x=dat[i][j];
		
                host_dat[i*new_npts+j].x=dat[i][j];
                host_dat[i*new_npts+j].y=0.0;
		}
		for(;j<new_npts;j++)
		{
                host_dat[i*new_npts+j].x=0.0;
                host_dat[i*new_npts+j].y=0.0;
		}
	}
	for(j=0;j<npts;j++)
	{
		host_stack[j].x=host_stack[j].x/ntrace;
		host_stack[j].y=0.0;
	}	
	for(j=npts;j<new_npts;j++)
	{
		host_stack[j].x=0.0;
                host_stack[j].y=0.0;
	}
	
	//cpy
	//comment it if new zero copy
	cudaMemcpy(dev_dat,host_dat,sizeof(cufftComplex)*ntrace*new_npts,cudaMemcpyHostToDevice);
	res=cufftPlanMany(&plan1,1,n1,NULL,1,new_npts,NULL,1,new_npts,CUFFT_C2C,ntrace);
	if(res!=0)
	{
	fprintf(stderr,"fail to create plan, error # %d\nCUFFT_INVALID_PLAN = 1\nCUFFT_ALLOC_FAILED = 2\nCUFFT_INVALID_TYPE = 3\nCUFFT_INVALID_VALUE = 4\nCUFFT_INTERNAL_ERROR = 5\nCUFFT_EXEC_FAILED = 6\nCUFFT_SETUP_FAILED = 7\nCUFFT_INVALID_SIZE = 8\nCUFFT_INCOMPLETE_PARAMETER_LIST = 10\n CUFFT_INVALID_DEVICE = 11\nCUFFT_PARSE_ERROR = 12\nCUFFT_NO_WORKSPACE = 13\n",res);	
	exit(-1);
	}
	cufftExecC2C(plan1,dev_dat,dev_dat,CUFFT_FORWARD);
	//for test
	//prepare w matrix
	gaussian_matrix(w_mat,k,freq_vec,new_npts);
	//cpy w and ft matrix to gpu
	cudaMemcpy(dev_w_mat,w_mat,sizeof(float)*new_npts*new_npts,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ft_mat,ft_mat,sizeof(float)*new_npts*new_npts,cudaMemcpyHostToDevice);
	//allocate s matrix
	s_mat=(cufftComplex*)malloc(sizeof(cufftComplex)*new_npts*new_npts);
	cudaMalloc((void**)&dev_s_mat,sizeof(cufftComplex)*new_npts*new_npts);
	cudaMalloc((void**)&dev_sum_s,sizeof(cufftComplex)*new_npts*new_npts);

	//s transform of each traces
	if(new_npts*new_npts < MAX)
	{
		nloop = 1;
	}	
	else
	{
		nloop = new_npts*new_npts/MAX;
	}
	batch = new_npts/nloop;
	shift=batch*new_npts;
	//allocate plan
	res=cufftPlanMany(&plan2, 1, n1, NULL, 1, new_npts, NULL, 1, new_npts, CUFFT_C2C,batch);
	if(res!=0)
	{
	fprintf(stderr,"fail to create stran plan, error # %d\nCUFFT_INVALID_PLAN = 1\nCUFFT_ALLOC_FAILED = 2\nCUFFT_INVALID_TYPE = 3\nCUFFT_INVALID_VALUE = 4\nCUFFT_INTERNAL_ERROR = 5\nCUFFT_EXEC_FAILED = 6\nCUFFT_SETUP_FAILED = 7\nCUFFT_INVALID_SIZE = 8\nCUFFT_INCOMPLETE_PARAMETER_LIST = 10\n CUFFT_INVALID_DEVICE = 11\nCUFFT_PARSE_ERROR = 12\nCUFFT_NO_WORKSPACE = 13\n",res);	
	exit(-1);
	}
	for(i=0;i<ntrace;i++)
	{

		copy_complex_matrix_shift<<<bk,th>>>(dev_s_mat,dev_dat+i*new_npts,new_npts);
		complex_mul_float_mat<<<blocks,threads>>>(dev_s_mat,dev_w_mat,new_npts);	
		//ifft
		for(j=0;j<nloop;j++)
		{
                	cufftExecC2C(plan2,dev_s_mat+j*shift,dev_s_mat+j*shift,CUFFT_INVERSE);
		}
		update_smat<<<blocks,threads>>>(dev_s_mat,dev_ft_mat,new_npts);

		if(i!=0)
		{
			sum_smat<<<blocks,threads>>>(dev_sum_s,dev_s_mat,new_npts);
		}
		else
		{
			copy_complex_matrix<<<blocks,threads>>>(dev_sum_s,dev_s_mat,new_npts);
		}

	}
	//final weight
	pws_wght_mat<<<blocks,threads>>>(dev_sum_s,new_npts,ntrace,pwr);
	t5=clock();
	//linear stacking stran
	res=cufftPlan1d(&plan0,new_npts,CUFFT_C2C,1);
	if(res!=0)
	{
	fprintf(stderr,"fail to create  plan0, error # %d\nCUFFT_INVALID_PLAN = 1\nCUFFT_ALLOC_FAILED = 2\nCUFFT_INVALID_TYPE = 3\nCUFFT_INVALID_VALUE = 4\nCUFFT_INTERNAL_ERROR = 5\nCUFFT_EXEC_FAILED = 6\nCUFFT_SETUP_FAILED = 7\nCUFFT_INVALID_SIZE = 8\nCUFFT_INCOMPLETE_PARAMETER_LIST = 10\n CUFFT_INVALID_DEVICE = 11\nCUFFT_PARSE_ERROR = 12\nCUFFT_NO_WORKSPACE = 13\n",res);	
	exit(-1);
	}
//new one
		cudaMemcpy(dev_stack,host_stack,sizeof(cufftComplex)*new_npts,cudaMemcpyHostToDevice);	
	cufftExecC2C(plan0,dev_stack,dev_stack,CUFFT_FORWARD);
	cudaMemcpy(host_stack,dev_stack,sizeof(cufftComplex)*new_npts,cudaMemcpyDeviceToHost);	
	//stran part
	copy_complex_matrix_shift<<<bk,th>>>(dev_s_mat,dev_stack,new_npts);

	//do W*S
	complex_mul_float_mat<<<blocks,threads>>>(dev_s_mat,dev_w_mat,new_npts);
	for(j=0;j<nloop;j++)
	{
                cufftExecC2C(plan2,dev_s_mat+j*shift,dev_s_mat+j*shift,CUFFT_INVERSE);
	}
	cudaMemcpy(s_mat,dev_s_mat,sizeof(cufftComplex)*new_npts*new_npts,cudaMemcpyDeviceToHost);
	//end output s stran fo stack
	//now we have stran of stacking trace in s_mat

	//stran_inv
	mul_wght_mat<<<blocks,threads>>>(dev_s_mat,dev_sum_s,new_npts);
	
	cudaMemcpy(s_mat,dev_s_mat,sizeof(cufftComplex)*new_npts*new_npts,cudaMemcpyDeviceToHost);
	//integrate over time axis to get sp
	for(i=0;i<new_npts;i++)//freq
	{
		host_stack[i].x=0.0;
		host_stack[i].y=0.0;
		for(j=0;j<new_npts;j++)//time
		{
		host_stack[i].x=host_stack[i].x+s_mat[i*new_npts+j].x;
		host_stack[i].y=host_stack[i].y+s_mat[i*new_npts+j].y;
		}
		if(isnan(cuCabsf(host_stack[i]))){host_stack[i].x=0;host_stack[i].y=0;}
	}
	host_stack[0].x=0.0;
	host_stack[0].y=0.0;

	//ifft
	//new one
	cudaMemcpy(dev_stack,host_stack,sizeof(cufftComplex)*new_npts,cudaMemcpyHostToDevice);
	cufftExecC2C(plan0,dev_stack,dev_stack,CUFFT_INVERSE);
	//cudaDeviceSynchronize();
//new one
	cudaMemcpy(host_stack,dev_stack,sizeof(cufftComplex)*new_npts,cudaMemcpyDeviceToHost);
	t7=clock();
	//output time costs

	time_cost=((double)(t5-t3))/CLOCKS_PER_SEC*1000;
	printf("time cost for setting up Gaussian matrix is :%f ms\n",time_cost);
	time_cost=((double)(t7-t2))/CLOCKS_PER_SEC*1000;
	printf("time cost for GPU pws stack is:%f ms\n",time_cost);
	//free plans
	cufftDestroy(plan0);
	cufftDestroy(plan1);
	cufftDestroy(plan2);
	//free other cuda 
	cudaFree(dev_stack);
	cudaFree(dev_dat);
	cudaFree(dev_s_mat);
	cudaFree(dev_w_mat);
	cudaFree(dev_ft_mat);
	cudaFree(dev_tmp);
	cudaFree(dev_sum_s);
	cudaFreeHost(host_dat);
	//cudaFreeHost(host_stack);
	//only real part is useful
	SACHEAD hd = sachdr(dt,new_npts,b);
	float *tmp=(float*)malloc(sizeof(float)*new_npts);
	for(i=0;i<new_npts;i++)
	{
		tmp[i]=host_stack[i].x/new_npts;
		if(isnan(tmp[i])) tmp[i]=0.0;
	}
	write_sac(argv[2],hd,tmp);
	cudaFreeHost(host_stack);


	// now do the linear stacking using GPU
	int thread_num, block_x, block_y;
	if (npts > MAX_TH) {
		thread_num = MAX_TH;
		block_x = npts/thread_num + 1;
	}
	else {
		thread_num = npts;
		block_x = 1;
	}
	block_y = ntrace;
	dim3 blocksize(block_y, block_x);
	
	// allocate cuda memory
	float *host_data;
	float *host_out;
	float *device_data;
	float *device_out;
	cudaHostAlloc((void **) &host_data, sizeof(float)*npts*ntrace, cudaHostAllocDefault);
	cudaHostAlloc((void **) &host_out, sizeof(float)*npts, cudaHostAllocDefault);
	cudaMalloc((void **) &device_data, sizeof(float)*npts*ntrace);
	cudaMalloc((void **) &device_out, sizeof(float)*npts);
	t1 = clock();
	// setup host data;
	for (i = 0; i < npts; i ++) {
		for (j = 0; j < ntrace; j ++) {
			host_data[j*npts+i] = dat[j][i];
		}
		host_out[i] = 0.0;
	}
	cudaMemset(device_out, 0, sizeof(float)*npts);
	
	cudaMemcpy(device_data,host_data, sizeof(float)*npts*ntrace, cudaMemcpyHostToDevice);
	
	//call kernel
	linear_stack<<<blocksize, thread_num>>>(device_data, device_out, npts);
	
	cudaMemcpy(host_out,device_out, sizeof(float)*npts, cudaMemcpyDeviceToHost);
	//for(i =0 ; i < npts; i ++) {
		//printf("%d %f\n", i, host_out[i]);
	//}
	//printf("%d %d %d\n", block_x, block_y, thread_num);
	SACHEAD hd1 = sachdr(dt,npts,b);
	t2 = clock();
        time_cost=((double)(t2-t1))/CLOCKS_PER_SEC*1000;
	printf("time cost for GPU linear stack is:%f ms\n", time_cost);
	write_sac(argv[3],hd1,host_out);
	cudaFree(host_data);
	cudaFree(host_out);
	cudaFree(device_data);
	cudaFree(device_out);
	free(dat);
	free(tmp);
	
	
}
void mul_wght(cufftComplex *a,cufftComplex *b,int n)
{
	int i;
	float tmpx,tmpy;
	for(i=0;i<n;i++)
	{
		tmpx=a[i].x*b[i].x;
		tmpy=a[i].y*b[i].x;
		a[i].x=tmpx;
		a[i].y=tmpy;
	}
}
void pws_wght(cufftComplex *s,int n,int ntr,int pwr)
{
	int i;
	float abs;
	for(i=0;i<n;i++)
	{
		abs=sqrt(s[i].x*s[i].x + s[i].y*s[i].y);
		s[i].x = abs*abs;
		}	
}
void sum_s_mat(cufftComplex *sum,cufftComplex *s,int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		sum[i].x=(sum[i].x + s[i].x);	
		sum[i].y=(sum[i].y + s[i].y);	
	}
}
void update_s_mat(cufftComplex *s,cufftComplex *ft,int n)
{
//  s/abs(s) * exp(2*pi*i*fvec*tvec)
	int i;
	float abs;
	float tmpx,tmpy;
	for(i=0;i<n;i++)
	{
		abs=sqrt(s[i].x*s[i].x + s[i].y*s[i].y);
		if(isnan(abs)) abs=1;
		tmpx=(s[i].x * ft[i].x - s[i].y*ft[i].y)/abs;	
		tmpy=(s[i].y * ft[i].x + s[i].x*ft[i].y)/abs;	
		s[i].x=tmpx;
		s[i].y=tmpy;
	}
}

//return 2^n >= a
int nextpow2(int a)
{
        int x;
	x=1;
        while(x<a)
        {
                x=x*2;		
	}
        return(x);
}
void gaussian_matrix(float *w,int k,float *fvec,int n)
{
        int i,j;
	float a,freq;
        float pi=3.14159;
	for(i=0;i<n;i++)
	{
	freq=fvec[i];
	for(j=0;j<n;j++)
	{
		a=fvec[j];
		w[i*n+j]=exp(-2*pi*pi*a*a/((k/2)*freq*freq));
	}
	}

}
void complex_mul_float(cufftComplex *a,float *b,int n)
{
        int i;
	for(i=0;i<n;i++)
	{
		a[i].x=a[i].x*b[i];
		a[i].y=a[i].y*b[i];
	}
}

void read_trace_list(char *list,float **dat,int *n,int *npts,float *dt,float *b)
{
	FILE *fp;
	SACHEAD hd0,hd1;
	
	int i;
	char line[200];
	char name[100];
	fp=fopen(list,"r");
	fgets(line,99,fp);
	sscanf(line,"%s",name);
	dat[0]=read_sac(name,&hd0);

	(*b) = hd0.b;
	(*dt) = hd0.delta;
	(*npts) = hd0.npts;
	i = 1;

	while(fgets(line,99,fp) != NULL)
	{
		sscanf(line,"%s",name);
		dat[i]=read_sac(name,&hd1);
		i=i+1;
		if(hd1.delta != hd0.delta || hd1.npts != hd0.npts)
		{
			fprintf(stderr,"Different dt or npts %s\n",name);
			i=i-1; 
		}
//		fprintf(stderr,"read %s\n",name);
	}
	(*n)=i;
}
