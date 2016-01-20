/* Cuda Code
 * for fitting an asymmetric PSF
 *
 */
#define BSZ 64				//!< max number of threads per block
#define MEM 524288			//!< maximum memory 524288
#define IMSZBIG 21			//!< maximum fitting window size
#define NK 128				//!< number of blocks to run in each kernel
#define pi  3.1415927f		//!< ensure a consistent value for pi
#define pi2 6.2831853f  	//!< ensure a consistent value for 2*pi
#define sq2pi 2.5066282f	//!< ensure value of sqrt of 2*pi
#define NV_P 4				//!< number of fitting parameters for MLEfit (x,y,bg,I)
#define NV_PS 5				//!< number of fitting parameters for MLEFit_sigma (x,y,bg,I,Sigma)
#define NV_PZ 5				//!< number of fitting parameters for MLEFit_z (x,y,bg,I,z)
#define NV_PS2 6			//!< number of fitting parameters for MLEFit_sigmaxy (x,y,bg,I,Sx,Sy)

//*******************************************************************************************
// Internal Calls
//*******************************************************************************************
__device__ void kernel_MatInvN(float * M, float * Minv, float * DiagMinv,
		int sz) {
	/*!
	 * \brief nxn partial matrix inversion
	 * \param M matrix to inverted
	 * \param Minv inverted matrix result
	 * \param DiagMinv just the inverted diagonal
	 * \param sz size of the matrix
	 */
	int ii, jj, kk, num, b;
	float tmp1 = 0;
	float yy[25];

	for (jj = 0; jj < sz; jj++) {
		//calculate upper matrix
		for (ii = 0; ii <= jj; ii++)
			//deal with ii-1 in the sum, set sum(kk=0->ii-1) when ii=0 to zero
			if (ii > 0) {
				for (kk = 0; kk <= ii - 1; kk++)
					tmp1 += M[ii + kk * sz] * M[kk + jj * sz];
				M[ii + jj * sz] -= tmp1;
				tmp1 = 0;
			}

		for (ii = jj + 1; ii < sz; ii++)
			if (jj > 0) {
				for (kk = 0; kk <= jj - 1; kk++)
					tmp1 += M[ii + kk * sz] * M[kk + jj * sz];
				M[ii + jj * sz] = (1 / M[jj + jj * sz])
						* (M[ii + jj * sz] - tmp1);
				tmp1 = 0;
			} else {
				M[ii + jj * sz] = (1 / M[jj + jj * sz]) * M[ii + jj * sz];
			}
	}

	tmp1 = 0;

	for (num = 0; num < sz; num++) {
		// calculate yy
		if (num == 0)
			yy[0] = 1;
		else
			yy[0] = 0;

		for (ii = 1; ii < sz; ii++) {
			if (ii == num)
				b = 1;
			else
				b = 0;
			for (jj = 0; jj <= ii - 1; jj++)
				tmp1 += M[ii + jj * sz] * yy[jj];
			yy[ii] = b - tmp1;
			tmp1 = 0;
		}

		// calculate Minv
		Minv[sz - 1 + num * sz] = yy[sz - 1] / M[(sz - 1) + (sz - 1) * sz];

		for (ii = sz - 2; ii >= 0; ii--) {
			for (jj = ii + 1; jj < sz; jj++)
				tmp1 += M[ii + jj * sz] * Minv[jj + num * sz];
			Minv[ii + num * sz] = (1 / M[ii + ii * sz]) * (yy[ii] - tmp1);
			tmp1 = 0;
		}
	}

	if (DiagMinv)
		for (ii = 0; ii < sz; ii++)
			DiagMinv[ii] = Minv[ii * sz + ii];

	return;
}

//*******************************************************************************************
__device__ float kernel_IntGauss1D(const int ii, const float x,
		const float sigma) {
	/*!
	 * \brief /f$ \frac{1}{2} /f$
	 * \param ii ???
	 * \param x ???
	 * \param sigma sigma value of the PSF
	 * \return float
	 */
	const float norm = 0.5f / sigma / sigma;
	return 0.5f
			* (erff((ii - x + 0.5f) * sqrtf(norm))
					- erff((ii - x - 0.5f) * sqrtf(norm)));
}

//*******************************************************************************************
__device__ float kernel_alpha(const float z, const float Ax, const float Bx,
		const float d) {
	/*!
	 * \brief compute coefficient for alpha
	 * \param z ???
	 * \param Ax ???
	 * \param Bx ???
	 * \param d ???
	 * \return float alpha value
	 */

	return 1.0f + __powf(z / d, 2) + Ax * __powf(z / d, 3) + Bx * __powf(z / d, 4);
}

//*******************************************************************************************
__device__ float kernel_dalphadz(const float z, const float Ax, const float Bx,
		const float d) {
	/*!
	 * \brief compute first derivative for alpha in relation to z
	 * \param z ???
	 * \param Ax ???
	 * \param Bx ???
	 * \param d ???
	 * \return float alpha value
	 */
	return (2.0f * z / (d * d) + 3.0f * Ax * __powf(z, 2) / (d * d * d)
			+ 4.0f * Bx * __powf(z, 3) / __powf(d, 4));
}

//*******************************************************************************************
__device__ float kernel_d2alphadz2(const float z, const float Ax,
		const float Bx, const float d) {
	/*!
	 * \brief compute second derivative for alpha in relation to z
	 * \param z ???
	 * \param Ax ???
	 * \param Bx ???
	 * \param d ???
	 * \return float alpha value
	 */
	return (2.0f / (d * d) + 6.0f * Ax * z / (d * d * d)
			+ 12.0f * Bx * __powf(z, 2) / __powf(d, 4));
}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss1D(const int ii, const float x,
		const float sigma, const float N, const float PSFy, float *dudt,
		float *d2udt2) {
	/*!
	 * \brief compute the derivative of the 1D gaussian
	 * \param ii ???
	 * \param x ???
	 * \param sigma ???
	 * \param N ???
	 * \param PSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */
	float a, b;
	a = expf(-0.5f * __powf(((ii + 0.5f - x) / sigma), 2.0f));
	b = expf(-0.5f * __powf((ii - 0.5f - x) / sigma, 2.0f));

	*dudt = -N / sq2pi / sigma * (a - b) * PSFy;

	if (d2udt2)
		*d2udt2 = -N / sq2pi / __powf(sigma, 3)
				* ((ii + 0.5f - x) * a - (ii - 0.5f - x) * b) * PSFy;
}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss1DSigma(const int ii, const float x,
		const float Sx, const float N, const float PSFy, float *dudt,
		float *d2udt2) {
	/*!
	 * \brief compute the derivative of the 1D gaussian
	 * \param ii ???
	 * \param x ???
	 * \param Sx ???
	 * \param N ???
	 * \param PSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */

	float ax, bx;

	ax = expf(-0.5f * powf(((ii + 0.5f - x) / Sx), 2.0f));
	bx = expf(-0.5f * powf((ii - 0.5f - x) / Sx, 2.0f));
	*dudt = -N / sq2pi / Sx / Sx
			* (ax * (ii - x + 0.5f) - bx * (ii - x - 0.5f)) * PSFy;

	if (d2udt2)
		*d2udt2 = -2.0f / Sx * dudt[0]
				- N / sq2pi / __powf(Sx, 5)
						* (ax * __powf((ii - x + 0.5f), 3)
								- bx * __powf((ii - x - 0.5f), 3)) * PSFy;
}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss2DSigma(const int ii, const int jj,
		const float x, const float y, const float S, const float N,
		const float PSFx, const float PSFy, float *dudt, float *d2udt2) {
	/*!
	 * \brief compute the derivative of the 2D gaussian
	 * \param ii ???
	 * \param jj ???
	 * \param x ???
	 * \param y ???
	 * \param S ???
	 * \param N ???
	 * \param PSFx ???
	 * \param PSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */

	float dSx, dSy, ddSx, ddSy;

	kernel_DerivativeIntGauss1DSigma(ii, x, S, N, PSFy, &dSx, &ddSx);
	kernel_DerivativeIntGauss1DSigma(jj, y, S, N, PSFx, &dSy, &ddSy);

	*dudt = dSx + dSy;
	if (d2udt2)
		*d2udt2 = ddSx + ddSy;

}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss2Dz(const int ii, const int jj,
		const float *theta, const float PSFSigma_x, const float PSFSigma_y,
		const float Ax, const float Ay, const float Bx, const float By,
		const float gamma, const float d, float *pPSFx, float *pPSFy,
		float *dudt, float *d2udt2) {
	/*!
	 * \brief compute the derivative of the 2D gaussian
	 * \param ii ???
	 * \param jj ???
	 * \param theta ???
	 * \param PSFSigma_x ???
	 * \param PSFSigma_y ???
	 * \param Ax ???
	 * \param Ay ???
	 * \param Bx ???
	 * \param By ???
	 * \param gamma ???
	 * \param d ???
	 * \param pPSFx ???
	 * \param pPSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */

	float Sx, Sy, dSx, dSy, ddSx, ddSy, dSdzx, dSdzy, ddSddzx, ddSddzy;
	float z, PSFx, PSFy, alphax, alphay, ddx, ddy;
	float dSdalpha_x, dSdalpha_y, d2Sdalpha2_x, d2Sdalpha2_y;
	z = theta[4];

	alphax = kernel_alpha(z - gamma, Ax, Bx, d);
	alphay = kernel_alpha(z + gamma, Ay, By, d);

	Sx = PSFSigma_x * sqrtf(alphax);
	Sy = PSFSigma_y * sqrtf(alphay);


	PSFx = kernel_IntGauss1D(ii, theta[0], Sx);
	PSFy = kernel_IntGauss1D(jj, theta[1], Sy);
	*pPSFx = PSFx;
	*pPSFy = PSFy;

	kernel_DerivativeIntGauss1D(ii, theta[0], Sx, theta[2], PSFy, &dudt[0], &ddx);
	kernel_DerivativeIntGauss1D(jj, theta[1], Sy, theta[2], PSFx, &dudt[1], &ddy);
	kernel_DerivativeIntGauss1DSigma(ii, theta[0], Sx, theta[2], PSFy, &dSx, &ddSx);
	kernel_DerivativeIntGauss1DSigma(jj, theta[1], Sy, theta[2], PSFx, &dSy, &ddSy);

	dSdalpha_x = PSFSigma_x / 2.0f / sqrtf(alphax);
	dSdalpha_y = PSFSigma_y / 2.0f / sqrtf(alphay);

	dSdzx = dSdalpha_x * kernel_dalphadz(z - gamma, Ax, Bx, d);
	dSdzy = dSdalpha_y * kernel_dalphadz(z + gamma, Ay, By, d);
	dudt[4] = dSx * dSdzx + dSy * dSdzy;

	if (d2udt2) {
		d2udt2[0] = ddx;
		d2udt2[1] = ddy;

		d2Sdalpha2_x = -PSFSigma_x / 4.0f / __powf(alphax, 1.5f);
		d2Sdalpha2_y = -PSFSigma_y / 4.0f / __powf(alphay, 1.5f);

		ddSddzx = d2Sdalpha2_x * __powf(kernel_dalphadz(z - gamma, Ax, Bx, d), 2)
				+ dSdalpha_x * kernel_d2alphadz2(z - gamma, Ax, Bx, d);
		ddSddzy = d2Sdalpha2_y * __powf(kernel_dalphadz(z + gamma, Ay, By, d), 2)
				+ dSdalpha_y * kernel_d2alphadz2(z + gamma, Ay, By, d);

		d2udt2[4] = ddSx * (dSdzx * dSdzx) + dSx * ddSddzx
				+ ddSy * (dSdzy * dSdzy) + dSy * ddSddzy;
	}
}

//*******************************************************************************************
__device__ void kernel_CenterofMass2D(const int sz, const float *data, float *x, float *y) {
	/*!
	 * \brief compute the 2D center of mass of a subregion
	 * \param sz nxn size of the subregion
	 * \param data subregion to search
	 * \param x x coordinate to return
	 * \param y y coordinate to return
	 */
    float tmpx=0.0f;
    float tmpy=0.0f;
    float tmpsum=0.0f;
    int ii, jj;

    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        tmpx+=data[sz*jj+ii]*ii;
        tmpy+=data[sz*jj+ii]*jj;
        tmpsum+=data[sz*jj+ii];
    }
    *x=tmpx/tmpsum;
    *y=tmpy/tmpsum;
}

//*******************************************************************************************
__device__ void kernel_GaussFMaxMin2D(const int sz, const float sigma, float * data, float *MaxN, float *MinBG) {
    /*!
	 * \brief returns filtered min and pixels of a given subregion
	 * \param sz nxn size of the subregion
	 * \param sigma used in filter calculation
	 * \param data the subregion to search
	 * \param MaxN maximum pixel value
	 * \param MinBG minimum background value
	 */
    int ii, jj, kk, ll;
    float filteredpixel=0, sum=0;
    float temp=0;
    *MaxN=0.0f;
    *MinBG=10e10f; //big

    float norm=0.5f/sigma/sigma;
    //loop over all pixels
    for (kk=0;kk<sz;kk++)
    	for (ll=0;ll<sz;ll++){
        filteredpixel=0.0f;
        sum=0.0f;
        for (ii=0;ii<sz;ii++)
        	for(jj=0;jj<sz;jj++){
        	temp=__expf(-pow((float)(ii-kk), 2)*norm)*__expf(-pow((float)(ll-jj), 2)*norm);
            filteredpixel+=temp*data[ii*sz+jj];
            sum+=temp;
        }
        filteredpixel/=sum;

        *MaxN=fmaxf(*MaxN, filteredpixel);
        *MinBG=fminf(*MinBG, filteredpixel);
    }
}

//***************************************************************************************************************************

__device__ void kernel_CentroidFitter(const int sz, const float *data, float *sx, float *sy,
		float *sx_std, float *sy_std){

	float tmpsx=0.0f; float tmpsx_std=0.0f;
    float tmpsy=0.0f; float tmpsy_std=0.0f;
    float tmpsum=0.0f; float tmpsum_std=0.0f;
    float min = 10000.0f;
    int ii , jj;
    float total = 0.0f;


    int center = (sz-1)/2;

    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
    total +=data[sz*jj+ii];
    }

    float thrsh = total/(sz*sz);

    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
    	if(data[sz*jj+ii]>thrsh){
        tmpsx+=data[sz*jj+ii]*ii;
        tmpsy+=data[sz*jj+ii]*jj;
        tmpsum+=data[sz*jj+ii];
    	}
    }

    *sx=tmpsx/tmpsum;
    *sy=tmpsy/tmpsum;

    if (*sx> (center+1) ){
    	*sx = center+1;
    }
    else if (*sx<(center-1)){
    	*sx = center-1;
    }


    if (*sy >(center+1)){
    	*sy = center+1;
    }
    else if (*sy <(center-1)){
    	*sy = center-1;
    }


    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
    	if(data[sz*jj+ii]<min){
    		min = data[sz*jj+ii];
    	}
    }

    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
    	if(data[sz*jj+ii]>thrsh){
    		tmpsum_std+=(data[sz*jj+ii]-min);
    		tmpsx_std+=(data[sz*jj+ii]-min)*(ii-*sx)*(ii-*sx);
    		tmpsy_std+=(data[sz*jj+ii]-min)*(jj-*sy)*(jj-*sy);
    	}
    }
    tmpsx_std/=tmpsum_std;
    tmpsy_std/=tmpsum_std;
    *sx_std=tmpsx_std;
    *sy_std=tmpsy_std;
    }


//*******************************************************************************************
// Global Calls
//*******************************************************************************************
extern "C"
__global__ void kernel_MLEFit(float *d_data, float PSFSigma, int sz, int iterations,
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits){
	/*!
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */
    //__shared__ float s_data[MEM];
    float M[NV_P*NV_P], Diag[NV_P], Minv[NV_P*NV_P];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=NV_P;
    float dudt[NV_P];
    float d2udt2[NV_P];
    float NR_Numerator[NV_P], NR_Denominator[NV_P];
    float theta[NV_P];
    float maxjump[NV_P]={1e0f, 1e0f, 1e2f, 2e0f};
    float gamma[NV_P]={1.0f, 1.0f, 0.5f, 1.0f};
    float Nmax;

    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;

    memset(M, 0, NV_P*NV_P*sizeof(float));
	memset(Minv, 0, NV_P*NV_P*sizeof(float));
	//load data
	float *s_data = d_data + (sz*sz*bx*BlockSize + sz*sz*tx);
	//initial values
	kernel_CenterofMass2D(sz, s_data, &theta[0], &theta[1]);
	kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &theta[3]);
    theta[2]=fmaxf(0.0f, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);

    for (kk=0;kk<iterations;kk++) {//main iterative loop

		//initialize
		memset(NR_Numerator, 0, NV_P*sizeof(float));
		memset(NR_Denominator, 0, NV_P*sizeof(float));

		for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
			PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
			PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);

			model=theta[3]+theta[2]*PSFx*PSFy;
			data = s_data[sz*jj + ii];

			//calculating derivatives
			kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
			kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
			dudt[2] = PSFx*PSFy;
			d2udt2[2] = 0.0f;
			dudt[3] = 1.0f;
			d2udt2[3] = 0.0f;

			cf=0.0f;
			df=0.0f;
			if (model>10e-3f) cf=data/model-1;
			if (model>10e-3f) df=data/pow(model, 2);
			cf=fminf(cf, 10e4f);
			df=fminf(df, 10e4f);

			for (ll=0;ll<NV;ll++){
				NR_Numerator[ll]+=dudt[ll]*cf;
				NR_Denominator[ll]+=d2udt2[ll]*cf-powf(dudt[ll], 2)*df;
			}
		}

		// The update
		if (kk<2)
			for (ll=0;ll<NV;ll++)
				theta[ll]-=gamma[ll]*fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
		else
			for (ll=0;ll<NV;ll++)
				theta[ll]-=fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

		// Any other constraints
		theta[2]=fmaxf(theta[2], 1.0f);
		theta[3]=fmaxf(theta[3], 0.01f);
    }

    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);

        model=theta[3]+theta[2]*PSFx*PSFy;
        data = s_data[sz*jj + ii];

        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0f;

        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }

        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*logf(model)-model-data*logf(data)+data;
            else
                Div+=-model;
    }

    // Matrix inverse (CRLB=F^-1) and output assignments
    kernel_MatInvN(M, Minv, Diag, NV);

    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;

    return;
}

//*******************************************************************************************
extern "C"
__global__ void kernel_MLEFit_sigma(float *d_data, float PSFSigma, int sz, int iterations,
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits){
	/*!
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */
    //__shared__ float s_data[MEM];
    float M[NV_PS*NV_PS], Diag[NV_PS], Minv[NV_PS*NV_PS];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=NV_PS;
    float dudt[NV_PS];
    float d2udt2[NV_PS];
    float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
    float theta[NV_PS];
    float maxjump[NV_PS]={1e0f, 1e0f, 1e2f, 2e0f, 5e-1f};
    float gamma[NV_PS]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f};
    float Nmax;

    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;

    memset(M, 0, NV_P*NV_P*sizeof(float));
	memset(Minv, 0, NV_P*NV_P*sizeof(float));
	//load data
	float *s_data = d_data + (sz*sz*bx*BlockSize + sz*sz*tx);

    //initial values
	kernel_CenterofMass2D(sz, s_data, &theta[0], &theta[1]);
	kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &theta[3]);
    theta[2]=fmaxf(0.0f, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;

    for (kk=0;kk<iterations;kk++) {//main iterative loop

        //initialize
		memset(NR_Numerator, 0, NV_P*sizeof(float));
		memset(NR_Denominator, 0, NV_P*sizeof(float));

        for (ii=0;ii<sz;ii++) 
			for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
            PSFy=kernel_IntGauss1D(jj, theta[1], theta[4]);

            model=theta[3]+theta[2]*PSFx*PSFy;
            data = s_data[sz*jj + ii];

            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0f;
            dudt[3] = 1.0f;
            d2udt2[3] = 0.0f;

            cf=0.0f;
            df=0.0f;
            if (model>10e-3f) cf=data/model-1;
            if (model>10e-3f) df=data/powf(model, 2);
            cf=fminf(cf, 10e4f);
            df=fminf(df, 10e4f);

            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-powf(dudt[ll], 2)*df;
            }
        }

        // The update
        if (kk<5)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=gamma[ll]*fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        // Any other constraints
        theta[2]=fmaxf(theta[2], 1.0f);
        theta[3]=fmaxf(theta[3], 0.01f);
        theta[4]=fmaxf(theta[4], 0.5f);
        theta[4]=fminf(theta[4], sz/2.0f);
    }

    // Calculating the CRLB and LogLikelihood
    Div=0.0f;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);

        model=theta[3]+theta[2]*PSFx*PSFy;
        data = s_data[sz*jj + ii];

        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0f;

        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }

        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*logf(model)-model-data*logf(data)+data;
            else
                Div+=-model;
    }

    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);


    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;

    return;
}

//*******************************************************************************************
extern "C"
__global__ void kernel_MLEFit_z(float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, float By, float gamma, float d, float PSFSigma_y, int sz, int iterations,
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits){
	/*!
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma_x sigma of the point spread function on the x axis
	 * \param Ax ???
	 * \param Ay ???
	 * \param Bx ???
	 * \param By ???
	 * \param gamma ???
	 * \param d ???
	 * \param PSFSigma_y sigma of the point spread function on the y axis
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */
    //__shared__
    float s_data[MEM];
    float M[5*5], Diag[5], Minv[5*5];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=5;
    float dudt[5];
    float d2udt2[5];
    float NR_Numerator[5], NR_Denominator[5];
    float theta[5];
    float maxjump[5]={1e0f, 1e0f, 1e2f, 2e0f, 1e-1f};
    float g[5]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f};
    float Nmax;
    float comx, comy, sigmax_sqrd, sigmay_sqrd;

    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;

    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

    //copy in data
    for (ii=0;ii<sz;ii++)
    	for(jj=0;jj<sz;jj++)
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];

    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma_x, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=fmaxf(0.0f, (Nmax-theta[3])*2*pi*PSFSigma_x*PSFSigma_y*sqrtf(2.0f));
    kernel_CentroidFitter(sz, &s_data[sz*sz*tx], &comx, &comy, &sigmax_sqrd, &sigmay_sqrd);
    float numerator = sigmay_sqrd- sigmax_sqrd;
    float numerator2 = 1/(4*gamma*PSFSigma_x*PSFSigma_y);
    theta[4]= powf(d,2)*numerator*numerator2;
    //theta[4]=0;

    for (kk=0;kk<iterations;kk++) {//main iterative loop

        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}

        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, dudt, d2udt2);

            model=theta[3]+theta[2]*PSFx*PSFy;
            data=s_data[sz*sz*tx+sz*jj+ii];

            //calculating remaining derivatives
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0f;
            dudt[3] = 1.0f;
            d2udt2[3] = 0.0f;

            cf=0.0f;
            df=0.0f;
            if (model>10e-3f) cf=data/model-1;
            if (model>10e-3f) df=data/pow(model, 2);
            cf=fminf(cf, 10e4f);
            df=fminf(df, 10e4f);

            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }

        // The update
        if (kk<2)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=g[ll]*fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        // Any other constraints
        theta[2]=fmaxf(theta[2], 1.0f);
        theta[3]=fmaxf(theta[3], 0.01f);

    }

    // Calculating the CRLB and LogLikelihood
    Div=0.0f;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {

        kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay, Bx,By, gamma, d, &PSFx, &PSFy, dudt, NULL);

        model=theta[3]+theta[2]*PSFx*PSFy;
        data=s_data[sz*sz*tx+sz*jj+ii];

        //calculating remaining derivatives
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0f;

        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }

        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*logf(model)-model-data*logf(data)+data;
            else
                Div+=-model;
    }

    // Matrix inverse (CRLB=F^-1)
    kernel_MatInvN(M, Minv, Diag, NV);

   //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
    return;
}

//*******************************************************************************************
extern "C"
__global__ void kernel_MLEFit_sigmaxy(float *d_data, float PSFSigma, int sz, int iterations,
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits){
	/*!
	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
	 * \param d_data array of subregions to fit copied to GPU
	 * \param PSFSigma sigma of the point spread function
	 * \param sz nxn size of the subregion to fit
	 * \param iterations number of iterations for solution to converge
	 * \param d_Parameters array of fitting parameters to return for each subregion
	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
	 * \param Nfits number of subregions to fit
	 */
    //__shared__
    float s_data[MEM];
    float M[6*6], Diag[6], Minv[6*6];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=6;
    float dudt[6];
    float d2udt2[6];
    float NR_Numerator[6], NR_Denominator[6];
    float theta[6];
    float maxjump[6]={1e0f, 1e0f, 1e2f, 2e0f, 1e-1f,1e-1f};
    float g[6]={1.0f, 1.0f, 0.5f, 1.0f, 1.0f,1.0f};
    float Nmax;

    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;

    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;

    //copy in data

    for (ii=0;ii<sz;ii++)
    	for(jj=0;jj<sz;jj++)
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];

    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=fmaxf(0.0f, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;
    theta[5]=PSFSigma;
    for (kk=0;kk<iterations;kk++) {//main iterative loop

        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}

        for (ii=0;ii<sz;ii++)
        	for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
            PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);

            model=theta[3]+theta[2]*PSFx*PSFy;
            data=s_data[sz*sz*tx+sz*jj+ii];

            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], &d2udt2[4]);
            kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], &d2udt2[5]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0f;
            dudt[3] = 1.0f;
            d2udt2[3] = 0.0f;

            cf=0.0f;
            df=0.0f;
            if (model>10e-3f) cf=data/model-1;
            if (model>10e-3f) df=data/pow(model, 2);
            cf=fminf(cf, 10e4f);
            df=fminf(df, 10e4f);

            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }

        // The update
            for (ll=0;ll<NV;ll++)
                theta[ll]-=g[ll]*fminf(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        // Any other constraints
        theta[2]=fmaxf(theta[2], 1.0f);
        theta[3]=fmaxf(theta[3], 0.01f);
        theta[4]=fmaxf(theta[4], PSFSigma/10.0f);
        theta[5]=fmaxf(theta[5], PSFSigma/10.0f);
    }

    // Calculating the CRLB and LogLikelihood
    Div=0.0f;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {

        PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
        PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);

        model=theta[3]+theta[2]*PSFx*PSFy;
        data=s_data[sz*sz*tx+sz*jj+ii];

        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
        kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0f;

        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }

        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*logf(model)-model-data*logf(data)+data;
            else
                Div+=-model;
    }

    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);

    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
    return;
}


