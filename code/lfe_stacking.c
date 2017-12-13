/*
CPU version of linear stacking and PWS stacking

Need to do FFT for PWS stacking

The sampling rate and number of points for all traces should be the same
*/
#include <fftw3.h>
#include <time.h>
#include "sacio.c"
#define NMAX 2048
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
void sum_s_matrix(fftw_complex *sum,fftw_complex *s,int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		sum[i][0] = sum[i][0]+s[i][0];
		sum[i][1] = sum[i][1]+s[i][1];
	}
}

void complex_mul_float(fftw_complex* a,float *b,int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		a[i][0] = a[i][0]*b[i];
		a[i][1] = a[i][1]*b[i];
	}

}
void update_s_mat(fftw_complex* a,fftw_complex *b,int n)
{
	int i;
	float abs;
	float tmpx,tmpy;
	for(i=0;i<n;i++)
	{
		abs=sqrt(a[i][0]*a[i][0]+a[i][1]*a[i][1]);
		if(isnan(abs)) abs=1;
		tmpx=(a[i][0] * b[i][0] - a[i][1]*b[i][1])/abs;
		tmpy=(a[i][1] * b[i][0] + a[i][0]*b[i][1])/abs;
		a[i][0] = tmpx;
		a[i][1] = tmpy;
	}

}

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
void gaussian_vector(float *w,int f,int k,float *fvec,int n)
{
	int i;
	float alpha,freq;;
	freq=fvec[f];
	float pi=3.14159;
	for(i=0;i<n;i++)
	{
		alpha=fvec[i];
		w[i]=exp(-2*pi*pi*alpha*alpha/((k/2)*freq*freq));
	}
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
		if(i==0){w[i*n+j]=0.0;};
	}									        }
}

void pws_wght(fftw_complex *sum,int n,int ntr,int pwr)
{
	int i;
	float abs;
	for(i=0;i<n;i++)
	{
		abs=sqrt(sum[i][0]*sum[i][0]+sum[i][1]*sum[i][1]);
//		sum[i][0] = pow(abs,pwr)/ntr;
		sum[i][0] = abs*abs/ntr;
	}
}
void mul_wght(fftw_complex *s,fftw_complex *sum,int n)
{
	//the wght is stored in sum[0]
	int i;
	for(i=0;i<n;i++)
	{
		s[i][0]=sum[i][0]*s[i][0];
		s[i][1]=sum[i][0]*s[i][1];
	}
}


// argv is the input_list gpu_out_name cpu_out_name
int main(int ac,char **argv)
{	
        if(ac!=4)
        {
		fprintf(stderr,"xxx input_list output_sac_pws output_sac_linear\n");
		exit(-1);							        }

	clock_t t1,t2, t3, t4, t5, t6;
	double time_cost;
	fftw_plan p1,p2;
	int npts,new_npts,ntrace;
	float dt,df,b;
	float *tvec,*fvec,*w;

	fftw_complex *in,*tmp,*s,*sum_s,*ft,*in_stack;
	float **dat;
	dat=(float**)malloc(sizeof(float*)*NMAX);
	float pi=3.14159,ft_tmp;
	int i,j;
	//t1 = clock();

	read_trace_list(argv[1],dat,&ntrace,&npts,&dt,&b);
	new_npts=nextpow2(npts);
	fprintf(stderr,"trace no: %d npts: %d dt: %f new_npts: %d\n",ntrace,npts,dt,new_npts);
	t1 =clock();	
	//allocate 
	w=(float*)malloc(new_npts*new_npts*sizeof(float));
	fvec=(float*)malloc(new_npts*sizeof(float));
	tvec=(float*)malloc(new_npts*sizeof(float));
	tmp=(fftw_complex*)fftw_malloc(2*new_npts*sizeof(fftw_complex));
	s=(fftw_complex*)fftw_malloc(new_npts*new_npts*sizeof(fftw_complex));
	in=(fftw_complex *)fftw_malloc(new_npts*ntrace*sizeof(fftw_complex));
	sum_s=(fftw_complex *)fftw_malloc(new_npts*new_npts*sizeof(fftw_complex));
	ft=(fftw_complex *)fftw_malloc(new_npts*new_npts*sizeof(fftw_complex));
	in_stack=(fftw_complex *)fftw_malloc(new_npts*sizeof(fftw_complex));
	
	
	df=1/((new_npts-1)*dt);
	//set up tvec and fvec
	for(i=0;i<(new_npts+1)/2;i++)
	{
		fvec[i]=i*df;
		tvec[i]=i*dt;
	}
	for(;i<new_npts;i++)
	{
		fvec[i]=(new_npts-i)*df;
		tvec[i]=i*dt;
	}
	for(i=0;i<new_npts;i++)
	{
		for(j=0;j<new_npts;j++)
		{
			ft_tmp=fvec[j]*tvec[i];
			ft[i*new_npts+j][0]=cos(ft_tmp*2*pi);
			ft[i*new_npts+j][1]=sin(ft_tmp*2*pi);
			//ft[j*new_npts+i][0]=cos(ft_tmp*2*pi);
			//ft[j*new_npts+i][1]=sin(ft_tmp*2*pi);
		}
	}
	//fprintf(stderr,"set up ft\n");
	//padding zeros;
	for(i=0;i<ntrace;i++)
	{
		for(j=0;j<npts;j++)
		{
			in[i*new_npts+j][0]=dat[i][j];
			in[i*new_npts+j][1]=0.0;
		}
		for(;j<new_npts;j++)
		{
			in[i*new_npts+j][0]=0.0;
			in[i*new_npts+j][1]=0.0;
		}
	}

	//fprintf(stderr,"ready to do  fft\n");
	
	// only record the time spent on processing traces without readin data

	//fft
	p1=fftw_plan_many_dft(1,&new_npts,ntrace,
			in,NULL,1,new_npts,
			in,NULL,1,new_npts,
			FFTW_FORWARD,FFTW_ESTIMATE);
	fftw_execute(p1);
	t3 = clock();
	//fprintf(stderr,"finish fft\n");
	//set up gaussian matrix
	gaussian_matrix(w,2,fvec,new_npts);
	for(i=0;i<ntrace;i++)
	{
		memcpy(tmp,in+i*new_npts,sizeof(fftw_complex)*new_npts);
		memcpy(tmp+new_npts,in+i*new_npts,sizeof(fftw_complex)*new_npts);
		for(j=0;j<new_npts;j++)
			memcpy(s+j*new_npts,tmp+j,sizeof(fftw_complex)*new_npts);
		complex_mul_float(s,w,new_npts*new_npts);
		//ifft
		p2=fftw_plan_many_dft(1,&new_npts,new_npts,
				s,NULL,1,new_npts,
				s,NULL,1,new_npts,
				FFTW_BACKWARD,FFTW_ESTIMATE);
		fftw_execute(p2);
		//update
		update_s_mat(s,ft,new_npts*new_npts);
		if(i!=0)
		{
			sum_s_matrix(sum_s,s,new_npts*new_npts);	
		}
		else
		{
			memcpy(sum_s,s,sizeof(fftw_complex)*new_npts*new_npts);
		}
	}	
	t4 = clock();
	//to pws wght
	pws_wght(sum_s,new_npts*new_npts,ntrace,2);
	
	//fprintf(stderr,"linear stacking\n");
	//linear stack
	for(i=0;i<npts;i++)
	{
		in_stack[i][1]=0.0;
		in_stack[i][0]=0.0;
	for(j=0;j<ntrace;j++)
	{
		in_stack[i][0]=in_stack[i][0]+dat[j][i];
	}
		in_stack[i][0]=in_stack[i][0]/ntrace;
	}		
	for(;i<new_npts;i++)
	{
		in_stack[i][1]=0.0;
		in_stack[i][0]=0.0;
	}
	t5 = clock();
	//fft 
	p1=fftw_plan_dft_1d(new_npts,in_stack,in_stack,FFTW_FORWARD,FFTW_ESTIMATE);
	fftw_execute(p1);
	//fprintf(stderr,"finish ls fft\n");
	t6 = clock();
	clock_t t7, t8, t9, t10;

	//stran part
	memcpy(tmp,in_stack,sizeof(fftw_complex)*new_npts);
	memcpy(tmp+new_npts,in_stack,sizeof(fftw_complex)*new_npts);

	for(i=0;i<new_npts;i++)
		memcpy(s+i*new_npts,tmp+i,sizeof(fftw_complex)*new_npts);

	complex_mul_float(s,w,new_npts*new_npts);
	t7 = clock();
	p1=fftw_plan_many_dft(1,&new_npts,new_npts,
			s,NULL,1,new_npts,
			s,NULL,1,new_npts,
			FFTW_BACKWARD,FFTW_ESTIMATE);
	fftw_execute(p1);
	t8 = clock();
	//fprintf(stderr,"got ls_stran\n");
	//mul w
	mul_wght(s,sum_s,new_npts*new_npts);
	//istran
	for(i=0;i<new_npts;i++)//freq
	{
	in_stack[i][0]=0.0;
	in_stack[i][1]=0.0;
	for(j=0;j<new_npts;j++)//time
	{
	in_stack[i][0]= in_stack[i][0]+s[i*new_npts+j][0];
	in_stack[i][1]= in_stack[i][1]+s[i*new_npts+j][1];
	}
	if(isnan(in_stack[i][0])){in_stack[i][0]=0.0;in_stack[i][1]=0.0;}
	}
	
	t9 = clock();
	p1=fftw_plan_dft_1d(new_npts,in_stack,in_stack,FFTW_BACKWARD,FFTW_ESTIMATE);
	fftw_execute(p1);
	t10 = clock();
	//write out
	SACHEAD hd=sachdr(dt,npts,b);
	float *ftmp=(float*)malloc(sizeof(float)*new_npts);
	for(i=0;i<new_npts;i++)
	{
		ftmp[i]=(float)(in_stack[i][0]);
		if(isnan(ftmp[i])){ftmp[i]=0.0;}
	}
	t2 = clock();
	write_sac(argv[2],hd,ftmp);
	//t2=clock();
	time_cost=((double)(t4-t3))/CLOCKS_PER_SEC*1000;
	fprintf(stderr,"time cost for setting up Gaussian matrix is :%f ms\n",time_cost);
	time_cost=((double)(t2-t1))/CLOCKS_PER_SEC*1000;
	fprintf(stderr,"time cost for CPU pws stack is :%f ms\n",time_cost);
	
	// now execute linear stacking for these traces
	t1 = clock();
	
	// data is stored in dat 2D array, length is npts and the number of traces is ntrace
	
	float *linear_stack = (float*)malloc(sizeof(float)*npts) ;
	// set up all values to be zero and stack
	for (i = 0; i < npts; i ++) {
		linear_stack[i] = 0.0;
		for (j = 0; j < ntrace; j ++){
			linear_stack[i] += dat[j][i];
		}	
		linear_stack[i] = linear_stack[i]/ntrace;
	}
	t2 = clock();
	time_cost=((double)(t2-t1))/CLOCKS_PER_SEC*1000;
	fprintf(stderr,"time cost for CPU linear stack is :%f ms\n",time_cost);
	write_sac(argv[3], hd, linear_stack);
	// write the sac out
	
	free(dat);
	free(ftmp);
	free(linear_stack);
	fftw_destroy_plan(p1);
	fftw_destroy_plan(p2);
	fftw_free(in);
	fftw_free(in_stack);
	fftw_free(s);
	fftw_free(sum_s);
	fftw_free(ft);
	fftw_free(tmp);
}
