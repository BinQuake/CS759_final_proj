/*
* cpu version of pws.
* Use fft package to do fast Fourier transformation
* Required packages:
* 1. fftw3: http://www.fftw.org fast Fourier transformation
* 2. gft: http://sourceforge.net/projects/fst-uofc/   fast S transformation
* 3. sacio: developed by Dr. Lupei Zhu http://www.eas.slu.edu/People/LZhu/home.htm
*
* run: pws_cpu list output.sac
* 
*/

#define NMAX 5000
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "sacio.c"
#include "fftw3.h"
#include "gft.c"

/*
 * Normalization of the current trace
 */
void cnor(double *x,double *xx)
{
	double nor;
	nor=(x[0]*x[0]+x[1]*x[1]);
	if(nor>0)
	{
	nor=sqrt(nor);
	xx[0]=x[0]/nor;
	xx[1]=x[1]/nor;
	}
	else
	{
		xx[0]=x[0];
		xx[1]=x[1];
	}
}

/*
 * b is the begin time
 * dt is the delta of sac trace which is sample rate
 */
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
		        }//check dt and npts
				
	            //check is nan
    	        if(isnan(dat[i-1][0]))
	            {
	                fprintf(stderr,"NaN trace %s\n",name);
	                i=i-1;
	            }
            }//end of while
															        (*n)=i;
}

// find the cloest power of 2 for int a
int power2( int a)
{
    int newa = 1;
    while (newa < a) 
        newa *= 2;
    return newa;
}

// complex number division
// x = a + ib
// y = c + id
int complexdiv(double *x, double *y) {
    double ac, bd, bc, ad;
    double ysquare = y[0]*y[0] + y[1]*y[1];

    ac = x[0]*y[0];
    bd = x[1]*y[1];
    bc = x[1]*y[0];
    ad = x[0]*y[1];

    if (ysquare > 0 ){
        x[0] = (ac+bd)/ysquare;
        x[1] = (bc-ad)/ysquare;
    }
    else {

        // there are some cases are padded with zeros
        x[0] = 0;
        x[1] = 0;
    }

}

// do the FFT for all traces sequentially
//
// for a single LFE event, all traces are concanated in sequential order
// *indx stores the starting position of each individual trace
// stride contains the number of points for each 
void inv_gft_1dComplex64(double *singal, unsigned int N, double *win, int *indx, int stride){

    int start, end, ct;
    double *band;
    int i;

    ct = 0;
    start = 0;
    while (indx[ct] >= 0) {
        end = indx[ct];
        
        // frequency band
        band = singal+ start*2*stride;
        
        // FFFT to transform to S-space
        fft(end-start, band, stride);
        start = indx[ct];
        ct ++; // go to the next trace
    }
    // remove windows
    for (i = 0; i < N*2; i +=2) {
        complexdiv(singal + i*stride, win+i);
    }
    ifft(N, singal, stride);
    return;
}

int main(int argc, char *argv) {

    if (argc != 3) {
        fprintf(stderr, "Usage: pws_cpu sac_file_list output_sac_file_name\n");
        exit(-1);
    }
    
    SACHEAD hdr;
    
    float **data;
    int npts, new_npts; // number of points for each trace and new pts after padding
    int num_trace;
    float b;
    float dt, df; // differential time and frequential
    
    clock_t t1, t2;

    // input signal
    int *indx;
    double *win;
    fftw_complex *in;
	fftw_complex *out;
	fftw_complex *out2;
	fftw_complex *wght;
	fftw_complex tmp;
    
    int i, j;
    data = float(**) malloc(sizeof(float*)*NMAX); // declare the whole array length
    
    /*
     *
     *Trace initialization
     *
     */
    
    // b is the begin time of each trace, not really useful here
    // dt is the sample rate should be all the same for all traces
    // npts should be all the same for all traces
    read_trace_list(argv[1], data, &num_trace, &npts, &dt, &b);
    new_npts = power2(npts);

    df = 1/(dt*(new_npts - 1)); // frequency is the  total time divided by 1

    in=(fftw_complex*)fftw_malloc(new_npts*ntrace*sizeof(fftw_complex));
	out=(fftw_complex*)fftw_malloc(new_npts*sizeof(fftw_complex));
	out2=(fftw_complex*)fftw_malloc(new_npts*sizeof(fftw_complex));
	wght=(fftw_complex*)fftw_malloc(new_npts*sizeof(fftw_complex));

	fprintf(stderr,"npts: %d  padding to %d\n",npts,new_npts);
	fprintf(stderr,"Total number of traces is %d\n",num_trace);

    // pad all traces with 0 in the end
    for (j =0; j < num_trace; j ++) {
        // jth trace
        for (i = 0; i < npts; i ++) {
            // ith point of jth trace
            in[i + j*new_npts][0] = data[j][i];
            in[i + j*new_npts][1] = 0.0; // setup original phase infor to be 0
        }
        for (i = npts; i < new_npts; i ++){
            in[i + j*new_npts][0] = in[i + j*new_npts][1] = 0.0;
        }
    }
    
    t1 = clock();
    /*
     * Now do the transformations
     */ 

    indx = gft_1dPartitions(new_npts);

    // gaussian is the window function
    win = windowsFromPars(new_npts, &gaussian, indx);

    // initialize output array
    for (i = 0; i < new_npts; i++){
        out[i][0] = out[i][1] = 0.0;
        wght[i][0] = wght[i][1] = 0.0;
    }
    
    for ( i = 0 ; i < num_trace; i ++) {
        // do the s transformation
        gft_1dComplex64(in + i*new_npts, new_npts, win, indx, 1);
        for (j = 0; j < new_npts; j ++) {
            // do the normalization and stacking
            out[j][0]=out[j][0]+in[i*new_npts+j][0];
    		out[j][1]=out[j][1]+in[i*new_npts+j][1];

    		// normalization of the input trace
	    	cnor(in[i*new_npts+j],tmp);
		    wght[j][0]=wght[j][0]+tmp[0];		
    		wght[j][1]=wght[j][1]+tmp[1];	
        }
    }
    
    // now after all stacking, add the weights
    for (j = 0; j < new_npts; j ++) {
        tmp[0]=sqrt(wght[j][0]*wght[j][0]+wght[j][1]*wght[j][1]);
		out2[j][0]=out[j][0]/ntrace;
		out2[j][1]=out[j][1]/ntrace;
		out[j][0]=out[j][0]*tmp[0]*tmp[0]/ntrace;
		out[j][1]=out[j][1]*tmp[0]*tmp[0]/ntrace;
    }
    
    // transform back to where it is
    inv_gft_1dComplex64(out,new_npts,win,par,1);
	inv_gft_1dComplex64(out2,new_npts,win,par,1);
    
    for(i=0;i<npts;i++)
	{
		data[0][i]=out[i][0]; //real part
		data[1][i]=out2[i][0]; //real part
	}
    
    // write to sac file
	hdr=sachdr(dt,npts,b);
	write_sac(argv[2],hdr,data[0]);
	write_sac("fst.tl.sac",hdr,data[1]);
	t2=clock();


	double time_cost=((double)(t2-t1))/CLOCKS_PER_SEC;
	fprintf(stderr,"time cost:%g s\n",time_cost);

	fftw_free(in);
	fftw_free(out);
	fftw_free(wght);
	free(win);
	free(par);
	free(data);
}
