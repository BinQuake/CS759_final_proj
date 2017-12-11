/*
 *	tf-pws based on fast s transform 
 *
 * zengxf, uw-madison, zengxf@geology.wisc.edu
 * 
 * ver 1. only supports BOX window function, 20151003
 * ver 2. fix a bug in cdiv, supports Gaussian window function, 20151016
 *
 * requires: gft, fftw3, sacio
 * gft:   http://sourceforge.net/projects/fst-uofc/
 * fftw3: http://www.fftw.org
 * sacio: developed by Dr. Lupei Zhu http://www.eas.slu.edu/People/LZhu/home.htm
 * l
 *
 * compile:  gcc tf_pws_fst.c -o tf_pws_fst -lfftw3 [-m32|-m64]
 * run:	     tf_pws_fst list output.sac
 *
 *
 * References: 
 * pws to LFE: Thurber et al., 2014, Phase-weighted stacking applied to low-frequency earthquakes. Bull. Seismol. Soc. Am., 104(5), 2567-2572
 * tf-pws: Schimmel M. and J. Gallart, 2007, Frequency-dependent phase coherence for noise suppression in seismic array data. J. Geophys. Res. 112, B04303
 * fast s transform: Brown R. et al., 2010, A general description of linear time-frequency transforms and formulation of a fast, invertible transform that samples the continuous S-transform spectrum nonredundantly. IEEE Transactions on Signal Processing, 58, 281-290.
 *
 *
 * License:  GNU General Public License version 3
 */
#define NMAX 5000
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "sacio.c"
#include "fftw3.h"
#include "gft.c"
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
void cdiv(double *x, double *y)
{
	double ac, bd, bc,ad;
	double	xx=y[0]*y[0]+y[1]*y[1];
	ac = x[0]*y[0];
	bd = x[1]*y[1];
	bc = x[1]*y[0];
	ad= x[0]*y[1];

	if(xx>0)
	{	x[0] = (ac+bd)/xx;
		x[1] = (bc-ad)/xx;
	}
	else{
		x[0]=0;
		x[1]=0;
	}
}

void igft_1dComplex64(double *signal, unsigned int N, double *win, int *pars, int stride)
{
	int fstart, fend, fcount;
	double *fband;
	int i;
	
	// Do the initial FFT of the signal
	//fft(N, signal, stride);
	// Apply the windows	
	//for (i=0; i<N*2; i+=2) {
	//	cdiv(signal+i*stride,win+i);
//	}
	// For each of the GFT frequency bands
	fcount = 0;
	fstart = 0;
	while (pars[fcount] >= 0) {
		fend = pars[fcount];
		// frequency band that we're working with
		fband = signal+fstart*2*stride;
		//  FFT to transform to S-space
		fft(fend-fstart, fband, stride);
		fstart = pars[fcount];
		fcount++;
	}
	//remove windows.
	for (i=0; i<N*2; i+=2) {
		cdiv(signal+i*stride,win+i);
	}
	ifft(N,signal,stride);
	return;
}

int main(int ac,char **av)
{
	if(ac != 3)
	{
		fprintf(stderr,"Usage: tf_pws_fst sac_file_list  output_sac file\n");
		exit(-1);
	}
	
	SACHEAD hd;
	float **dat;
	int new_npts;
	int ntrace;
	float b;
	int npts;
	float dt,df;
	clock_t t1,t2;
	//input signal
	//partions
	int *par;
	double *win;
	fftw_complex *in;
	fftw_complex *out;
	fftw_complex *out2;
	fftw_complex *wght;
	fftw_complex tmp;
	int i,j;
	dat=(float**)malloc(sizeof(float*)*NMAX);
	t1=clock();
	read_trace_list(av[1],dat,&ntrace,&npts,&dt,&b);
	new_npts=nextpow2(npts);

	df=1/(dt*(new_npts-1));
	//in=(double**)malloc(new_npts*sizeof(double*));
	//for(i=0;i<new_npts;i++){in[i]=(double*)malloc(2*sizeof(double));}
	in=(fftw_complex*)fftw_malloc(new_npts*ntrace*sizeof(fftw_complex));
	out=(fftw_complex*)fftw_malloc(new_npts*sizeof(fftw_complex));
	out2=(fftw_complex*)fftw_malloc(new_npts*sizeof(fftw_complex));
	wght=(fftw_complex*)fftw_malloc(new_npts*sizeof(fftw_complex));
	
	fprintf(stderr,"npts: %d  padding to %d\n",npts,new_npts);
	fprintf(stderr,"%d traces\n",ntrace);
	for(j=0;j<ntrace;j++)
	{
	for(i=0;i<npts;i++) {
		in[i+j*new_npts][0]=dat[j][i];
		in[i+j*new_npts][1]=0.0;
		}
	for(i=npts;i<new_npts;i++){in[i+j*new_npts][0]=in[i+j*new_npts][1]=0.0;}
	}

	par=gft_1dPartitions(new_npts);
	//now only support box windows function, so the result is same as DOST
	win=windowsFromPars(new_npts,&gaussian,par);
	for(i=0;i<new_npts;i++)
	{
	out[i][0]=out[i][1]=0.0;
	wght[i][0]=wght[i][1]=0.0;
	}

	for(i=0;i<ntrace;i++)
	{
	gft_1dComplex64(in+i*new_npts,new_npts,win,par,1);
	for(j=0;j<new_npts;j++)
	{
		out[j][0]=out[j][0]+in[i*new_npts+j][0];
		out[j][1]=out[j][1]+in[i*new_npts+j][1];
		cnor(in[i*new_npts+j],tmp);
		wght[j][0]=wght[j][0]+tmp[0];		
		wght[j][1]=wght[j][1]+tmp[1];		
	}	
	}
	for(j=0;j<new_npts;j++)
	{
		tmp[0]=sqrt(wght[j][0]*wght[j][0]+wght[j][1]*wght[j][1]);
		out2[j][0]=out[j][0]/ntrace;
		out2[j][1]=out[j][1]/ntrace;
		out[j][0]=out[j][0]*tmp[0]*tmp[0]/ntrace;
		out[j][1]=out[j][1]*tmp[0]*tmp[0]/ntrace;
		
	}

	igft_1dComplex64(out,new_npts,win,par,1);
	igft_1dComplex64(out2,new_npts,win,par,1);
	for(i=0;i<npts;i++)
	{
		dat[0][i]=out[i][0]; //real part
		dat[1][i]=out2[i][0]; //real part
	}
	hd=sachdr(dt,npts,b);
	write_sac(av[2],hd,dat[0]);
	write_sac("fst.tl.sac",hd,dat[1]);
	t2=clock();
	double time_cost=((double)(t2-t1))/CLOCKS_PER_SEC;
	        fprintf(stderr,"time cost:%g s\n",time_cost);

	fftw_free(in);
	fftw_free(out);
	fftw_free(wght);
	free(win);
	free(par);
	free(dat);

}
