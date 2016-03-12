#define MEM 524288			//!< max memory 524288
#define PI  3.141592654f	//!< ensure a consistent value for pi
#define SQRTPI	1.77245385091f
#define NV 4
#define POS_EPSILON 1.0E-04 // nm
#define INT_EPSILON 1.0E-04 // %/100
#define WID_EPSILON	1.0E-04 // nm
#define MAX_WIDTH 4.0f //
#define MIN_WIDTH 0.8f //photons
#define MIN_NOISE 1.0
#define MAX_NOISE 2.0

//! not defined in the C standard used by visual studio
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

__device__ void centerofMass(const int sz, const float *data, float *x, float *y) {
	float tmpx = 0.0f;
	float tmpy = 0.0f;
	float tmpsum = 0.0f;
	int ii, jj;
	for (jj = 0; jj<sz; jj++)
		for (ii = 0; ii<sz; ii++){
			tmpx += data[sz*jj + ii] * ii;
			tmpy += data[sz*jj + ii] * jj;
			tmpsum += data[sz*jj + ii];
		}
	*x = tmpx / tmpsum;
	*y = tmpy / tmpsum;
}

__device__ float minArray(float* array) {
	float min = array[0];
    size_t n = sizeof array / sizeof *array;
    for (size_t i = 1; i < n; i++) {
        if (array[i] < min) 
            min = array[i];
    }
    return min;
}

__device__ float maxArray(float* array) {
	float max = array[0];
    size_t n = sizeof array / sizeof *array;
    for (size_t i = 1; i < n; i++) {
        if (array[i] > max) 
            max = array[i];
    }
    return max;
}

__device__ float percentDifference(float a, float b) {
    float c = 2. * (a - b) / (a + b);
    return (c < 0.) ? -c : c;
}

__device__ float computeLogLikelihood(float* xsignal, float* parameters, float L, float k, float pixelsize, float usablepixel){
	float logLikelihood = 0.0f;
	float x, expected;
	float a = pixelsize*usablepixel;
	float x0 = parameters[0];
	float I0 = parameters[1];
	float bg = parameters[2];	
	float w = parameters[3];//0.75 * pi/(pixelsize*k);
        
    for (int n = 0; n < L; n++) {
		x = pixelsize*n + pixelsize/2;
        expected = bg*L + (I0*L*PI*w*w*(-erff((k*(-a/2 + x - x0))/w) + erff((k*( a/2 + x - x0))/w))) / (2*k*k);
        logLikelihood += xsignal[n]*log(expected) - expected;
    }
    return logLikelihood;
}

__device__ void getFirstDerivatives(float* array, float x, float* parameters, float L, float k, float pixelsize, float usablepixel) {
        
    float x0 = parameters[0];
	float I0 = parameters[1];
	float w = parameters[3];
    float a = pixelsize*usablepixel;
    
    // optimizations
    float k2 = k*k;
    float w2 = w*w;
    float y1 = a/2 + x - x0;
    float y2 = -a/2 + x - x0;
    float y1p2 = y1*y1;
    float y2p2 = y2*y2;
    
    array[0] = (I0*L*PI*((2*k)/(expf((k2*y2p2)/w2)*SQRTPI*w) - 
                    (2*k)/(expf((k2*y1p2)/w2)*SQRTPI*w))*w2)/(2*k2);
    array[1] = (L*PI*w2*(-erff((k*(y2))/w) 
                   + erff((k*(y1))/w)))/(2*k2);
    array[2] = L;
    array[3] = (I0*L*PI*w2*((2*k*(y2))/(expf((k2*y2p2)/w2)*SQRTPI*w2) - 
                       (2*k*(y1))/(expf((k2*y1p2)/w2)*SQRTPI*w2)))/(2*k2) + 
          (I0*L*PI*w*(-erff((k*(y2))/w) 
                     + erff((k*(y1))/w)))/k2;
}

__device__ void getSecondDerivatives(float* array, float x, float* parameters, float L, float k, float pixelsize, float usablepixel) {
    
    float x0 = parameters[0];
	float I0 = parameters[1];
	float w = parameters[3];
    float a = pixelsize*usablepixel;
    
    // optimizations
    float k2 = k*k;
    float k3 = k2*k;
    float w2 = w*w;
    float w3 = w2*w;
    float w5 = w2*w3;
    float y1 = a/2. + x - x0;
    float y2 = -a/2. + x - x0;
    float y1p2 = y1*y1;
    float y1p3 = y1*y1p2;
    float y2p2 = y2*y2;
    float y2p3 = y2*y2p2;
   
    // These are not meant to be human readable. They were automatically generated and then further optimized.
    array[0] = (I0*L*PI*w2*((4*k3*(y2))/(expf((k2*y2p2)/w2)*SQRTPI*w3) - 
                 (4*k3*(y1))/(expf((k2*y1p2)/w2)*SQRTPI*w3)))/(2*k2);
    array[1] = 0;
    array[2] = 0;
    array[3] = (I0*L*(4*PI*w*((2*k*(y2))/(expf((k2*y2p2)/w2)*SQRTPI*w2) - 
                   (2*k*(y1))/(expf((k2*y1p2)/w2)*SQRTPI*w2)) + 
           PI*w2*((-4*k*(y2))/(expf((k2*y2p2)/w2)*SQRTPI*w3) + 
                  (4*k3*y2p3)/(expf((k2*y2p2)/w2)*SQRTPI*w5) + 
                   (4*k*(y1))/(expf((k2*y1p2)/w2)*SQRTPI*w3) - 
                  (4*k3*y1p3)/(expf((k2*y1p2)/w2)*SQRTPI*w5)) + 
           2*PI*(-erff((k*(y2))/w) 
                + erff((k*(y1))/w))))/(2*k2);
}

__device__ void computeNewtonRaphson(float* signal, float* parameters, float* delta, float L, float k, float pixelsize, float usablepixel) {
        
    // allocate first and second derivates of loglikelihood
	float firstL[NV];
	float secondL[NV];
	float first[NV];
	float second[NV];
	memset(firstL, 0, NV*sizeof(float));
	memset(secondL, 0, NV*sizeof(float));
	float x, expected;
	int i,j,n;
	float a = pixelsize*usablepixel;
	float x0 = parameters[0];
	float I0 = parameters[1];
	float bg = parameters[2];	
	float w = parameters[3];
        
    // compute the log-likelihood derivatives
    for (n = 0; n < L; n++) {
		memset(first, 0, NV*sizeof(float));
		memset(second, 0, NV*sizeof(float));
		x = pixelsize*n + pixelsize/2;
        expected = bg*L + (I0*L*PI*w*w*(-erff((k*(-a/2 + x - x0))/w) + erff((k*( a/2 + x - x0))/w))) / (2*k*k);
        getFirstDerivatives(first, x, parameters, L, k, pixelsize, usablepixel);
        getSecondDerivatives(second, x, parameters, L, k, pixelsize, usablepixel);
    
        for (i = 0; i < NV; i++) {
            firstL[i]  += ((signal[n]/expected - 1)*first[i]);
            secondL[i] += ((signal[n]/expected - 1)*second[i] - (signal[n]*first[i]*first[i]/(expected*expected)));
        }
    }
     // compute the recommended change in parameter values   
	for (j = 0; j < NV; j++) 
    	delta[j] = firstL[j]/secondL[j];
}

__device__ bool doIteration(float* signal, float* parameters, float* delta, float* likelihood, float length, float wavenumber, float pixelsize, float usablepixel, float initialnoise){
	int k,j;
	float newlikelihood;    	

    // compute the Newton-Raphson parameter change
    computeNewtonRaphson(signal, parameters, delta, length, wavenumber, pixelsize, usablepixel);

    // coefficient for decaying update
    double coefficient = 1.0;
    
    float newparameters[NV];
	memset(newparameters, 0, NV*sizeof(float));
    
    // loop an arbitrary 10 times (will hopefully only iterate once)
    // This loop will continue as long as we have a smaller likelihood.
    // This assumes (with certainty) that the Newton-Raphson method overshoots.
    for (k = 0; k < 10; k++) {
        // update the new parameters
		for (j = 0; j < NV; j++) 
        	newparameters[j] = parameters[j]-delta[j]*coefficient;
        
        // ensure that the parameters are within bounds
        if (newparameters[2] < MIN_NOISE) 
            newparameters[2] = MIN_NOISE;
        else if (newparameters[2] > MAX_NOISE*initialnoise) 
            newparameters[2] = MAX_NOISE*initialnoise;
        
        // find the new log-likelihood
        newlikelihood = computeLogLikelihood(signal, newparameters, length, wavenumber, pixelsize, usablepixel);
    
        // end the loop if the likelihood increases; otherwise, halve coefficient
        if (newlikelihood > *likelihood) {
            
            // Oh good! Lets stop.
            *likelihood = newlikelihood;
            break;
        } else {
            // The likelihood decreased! Weird...  Try again.
            coefficient /= 2.0;
        }
    }
    
    // check end-conditions
    bool isDone = false;
    
    if (((delta[0] < 0) ? -delta[0] : delta[0]) < POS_EPSILON)
        isDone = true;
    else if (percentDifference(parameters[1], newparameters[1]) < INT_EPSILON)
        isDone = true;
    else if (((delta[3] < 0)? -delta[3] : delta[3]) < WID_EPSILON)
        isDone = true;
   
    // update the final parameters
    if (!isDone) 
		for (j = 0; j < NV; j++) 
        	parameters[j] = newparameters[j];
    
    return isDone;
}

__device__ bool isInvalid(float* param) {
    return isinf(param[0])   || isnan(param[0])   ||
           isinf(param[1])  || isnan(param[1])  ||
           isinf(param[2]) || isnan(param[2]) ||
           isinf(param[3])  || isnan(param[3]) ||
           param[1] < 0.0 || param[2] < 0.0 || param[3] < 0.0;
}

__device__ float getPartialExpected(float x, float* parameters, float L, float k, float pixelsize, float usablepixel){
	float a = pixelsize*usablepixel;
	float x0 = parameters[0];
	float w = parameters[3];
	return (L*PI*w*w*(-erff((k*(-a/2 + x - x0))/w) 
                         + erff((k*( a/2 + x - x0))/w)))/(2*k*k);
}		

__device__ void getPartialExpectedArray(float* expected, float* parameters, float windowsize, float wavenumber, float pixelsize, float usablepixel){		
	float x;	
	for(int n=0; n<windowsize; n++){
		x = pixelsize*n + pixelsize/2;
		expected[n]=getPartialExpected(x, parameters, windowsize, wavenumber, pixelsize, usablepixel);
	}	
}

extern "C"
__global__ void kernel_M2LEFit(float *d_data, int sz, float pixelsize, float usablepixel, int maxIter, float wavenumber, float *d_Parameters, int Nfits){

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BlockSize = blockDim.x;	
	float cx,cy,S;

	//load data
	float *s_data = d_data + (sz*sz*bx*BlockSize + sz*sz*tx);

	centerofMass(sz, s_data, &cx, &cy);
	int left   = max(0, cx - 3);
    int right  = min(sz, cx + 4);
    int top    = max(0, cy - 3);
    int bottom = min(sz, cy + 4);
    int width  = right - left;
    int height = bottom - top;
	if (width < 4 || height < 4) return;
	float* xsignal = new float[width];
	float* ysignal = new float[height];
	float xparam[NV];
	float yparam[NV];
	int kk,x,y;

	//memset(xsignal, 0, width*sizeof(float));
	//memset(ysignal, 0, height*sizeof(float));

	// accumulate pixel signal
	for (y = top; y < bottom; y++)
    	for (x = left; x < right; x++){
			S = s_data[sz*y + x];
            xsignal[x-left] += S;
            ysignal[y-top] += S;
        }

	// estimate the initial noise level
	float xnoise=minArray(xsignal)/width;
	float ynoise=minArray(ysignal)/height;
    float initialnoise = (xnoise + ynoise)/2.;

	float* expectedX = new float[width];
	float* expectedY = new float[height];

	// get initial estimates
	xparam[0]=cx;
	xparam[2]=xnoise;
	xparam[3]=2.3456387388762832235657480556918/(pixelsize*wavenumber);
	yparam[0]=cy;
	yparam[2]=ynoise;
    yparam[3]=2.3456387388762832235657480556918/(pixelsize*wavenumber);
	getPartialExpectedArray(expectedX, xparam, width, wavenumber, pixelsize, usablepixel);
	getPartialExpectedArray(expectedY, yparam, height, wavenumber, pixelsize, usablepixel);
	xparam[1]=(maxArray(xsignal)-minArray(xsignal))/maxArray(expectedX);
	yparam[1]=(maxArray(ysignal)-minArray(ysignal))/maxArray(expectedY);

	// initial likelihood calculations
    float xlikelihood = computeLogLikelihood(xsignal, xparam, height, wavenumber, pixelsize, usablepixel);
    float ylikelihood = computeLogLikelihood(ysignal, yparam, width, wavenumber, pixelsize, usablepixel);

 	// used for the change in parameters
    float delta[NV];
	memset(delta,0,NV*sizeof(float));
    
    // condition flags
    bool xdone = false;
    bool ydone = false;

	// update parameters
    for (int iter = 0; iter < maxIter; iter++) {
		// do iteration until done
        if (!xdone) xdone = doIteration(xsignal,
                                xparam, delta, &xlikelihood,
                                height, wavenumber, 
                                pixelsize, usablepixel, 
                                initialnoise);
        
        
        // do iteration until done
        if (!ydone) ydone = doIteration(ysignal,
                                yparam, delta, &ylikelihood,
                                width, wavenumber, 
                                pixelsize, usablepixel, 
                                initialnoise);
        
        
        // end the loop when both are done
        if (xdone && ydone) break;
	}

	// check for invalid parameters (to reject)
    if (isInvalid(xparam) || isInvalid(yparam)) 
        return;
    
    // check that the position estimate is within the region
    if (xparam[0] < 0 || xparam[0] > pixelsize*width ||
            yparam[0] < 0 || yparam[0] > pixelsize*height) 
        return;
  
    // check that the width of the PSF is within acceptable bounds
    if (xparam[3] < MIN_WIDTH || xparam[3] > MAX_WIDTH
            || yparam[3] < MIN_WIDTH || yparam[3] > MAX_WIDTH) 
        return;
    
	float theta[NV*2];
    // record information
    theta[0]=xparam[0]/pixelsize + left;
    theta[1]=yparam[0]/pixelsize + top;
    theta[2]=xparam[1];
    theta[3]=yparam[1];
    theta[4]=xparam[2];
    theta[5]=yparam[2];
    theta[6]=xparam[3];
    theta[7]=yparam[3];
    
	//write to global arrays
	for (kk = 0; kk<NV*2; kk++)
		d_Parameters[Nfits*kk + BlockSize*bx + tx] = theta[kk];
}
