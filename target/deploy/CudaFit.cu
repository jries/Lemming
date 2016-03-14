#define MEM 524288		//!< max memory 524288
#define IMSZBIG 21		//!< maximum fitting window size
#define pi  3.141592654f	//!< ensure a consistent value for pi
#define pi2 6.283185307f  	//!< ensure a consistent value for 2*pi
#define sq2pi 2.506628275f	//!< ensure value of sqrt of 2*pi
//#define two_sqrtpi 1.12837916f// 2/sqrt(pi)
//#define one_sqrtpi 0.564189584f // 1/sqrt(pi)
//#define rel_error 0.00000000001f      //calculate 12 significant figures
#define NV_P 4			//!< number of fitting parameters for MLEfit (x,y,bg,I)
#define NV_PS 5			//!< number of fitting parameters for MLEFit_sigma (x,y,bg,I,Sigma)
#define NV_PZ 5			//!< number of fitting parameters for MLEFit_z(x,y,bg,I,z)
#define NV_P2 6			//!< number of fitting parameters for MLEFit_sigmaxy (x,y,bg,I,Sx,Sy)

//__device__ float __fsqrt_rd(float  x);
//__device__ float __powf(float  x, float  y);
//__device__ float __expf(float  x);

#ifndef max
//! not defined in the C standard used by visual studio
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
//! not defined in the C standard used by visual studio
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

//*******************************************************************************************
// Internal Calls
//*******************************************************************************************

__device__ void kernel_MatInvN(float * M, float * Minv, float * DiagMinv, int sz) {

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
			}
			else {
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
__device__ float kernel_IntGauss1D(const int ii, const float x, const float sigma) {

	const float norm = 0.5f / sigma / sigma;
	return 0.5f
		* (erff((ii - x + 0.5f) * sqrtf(norm))
		- erff((ii - x - 0.5f) * sqrtf(norm)));
}

//*******************************************************************************************
__device__ float kernel_alpha(const float z, const float Ax, const float Bx, const float d) {
	float q = z / d;
	return 1.0f + (q*q) + Ax * (q*q*q) + Bx * (q*q*q*q);
}

//*******************************************************************************************
__device__ float kernel_dalphadz(const float z, const float Ax, const float Bx, const float d) {
	return (2.0f * z / (d*d) + 3.0f * Ax * (z*z) / (d*d*d)
		+ 4.0f * Bx * (z*z*z) / (d*d*d*d));
}

//*******************************************************************************************
__device__ float kernel_d2alphadz2(const float z, const float Ax, const float Bx, const float d) {
	return (2.0f / (d*d) + 6.0f * Ax * z / (d*d*d)
		+ 12.0f * Bx * (z*z) / (d*d*d*d));
}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss1D(const int ii, const float x,
	const float sigma, const float N, const float PSFy, float *dudt,
	float *d2udt2) {

	float a, b;
	a = expf(-0.5f * ((ii + 0.5f - x) / sigma)*((ii + 0.5f - x) / sigma));
	b = expf(-0.5f * ((ii - 0.5f - x) / sigma)*((ii - 0.5f - x) / sigma));

	*dudt = -N / sq2pi / sigma * (a - b) * PSFy;

	if (d2udt2)
		*d2udt2 = -N / sq2pi / (sigma*sigma*sigma)
		* ((ii + 0.5f - x) * a - (ii - 0.5f - x) * b) * PSFy;
}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss1DSigma(const int ii, const float x,
	const float Sx, const float N, const float PSFy, float *dudt,
	float *d2udt2) {

	float ax, bx;

	ax = expf(-0.5f * ((ii + 0.5f - x) / Sx)*((ii + 0.5f - x) / Sx));
	bx = expf(-0.5f * ((ii - 0.5f - x) / Sx)*((ii - 0.5f - x) / Sx));
	*dudt = -N / sq2pi / Sx / Sx
		* (ax * (ii - x + 0.5f) - bx * (ii - x - 0.5f)) * PSFy;

	if (d2udt2)
		*d2udt2 = -2.0f / Sx * dudt[0]
		- N / sq2pi / (Sx*Sx*Sx*Sx*Sx)
		* (ax * ((ii - x + 0.5f)*(ii - x + 0.5f)*(ii - x + 0.5f))
		- bx * ((ii - x - 0.5f)*(ii - x - 0.5f)*(ii - x - 0.5f))) * PSFy;
}

//*******************************************************************************************
__device__ void kernel_DerivativeIntGauss2DSigma(const int ii, const int jj,
	const float x, const float y, const float S, const float N,
	const float PSFx, const float PSFy, float *dudt, float *d2udt2) {

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

		d2Sdalpha2_x = -PSFSigma_x / 4.0f / powf(alphax, 1.5f);
		d2Sdalpha2_y = -PSFSigma_y / 4.0f / powf(alphay, 1.5f);

		ddSddzx = d2Sdalpha2_x * powf(kernel_dalphadz(z - gamma, Ax, Bx, d), 2)
			+ dSdalpha_x * kernel_d2alphadz2(z - gamma, Ax, Bx, d);
		ddSddzy = d2Sdalpha2_y * powf(kernel_dalphadz(z + gamma, Ay, By, d), 2)
			+ dSdalpha_y * kernel_d2alphadz2(z + gamma, Ay, By, d);

		d2udt2[4] = ddSx * (dSdzx * dSdzx) + dSx * ddSddzx
			+ ddSy * (dSdzy * dSdzy) + dSy * ddSddzy;
	}
}

//*******************************************************************************************
__device__ void kernel_CenterofMass2D(const int sz, const float *data, float *x, float *y) {

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

//*******************************************************************************************
__device__ void kernel_GaussFMaxMin2D(const int sz, const float sigma, float * data, float *MaxN, float *MinBG) {

	int ii, jj, kk, ll;
	float filteredpixel = 0, sum = 0;
	float temp = 0;
	*MaxN = 0.0f;
	*MinBG = 10e10f; //big

	float norm = 0.5f / sigma / sigma;
	//loop over all pixels
	for (kk = 0; kk<sz; kk++)
		for (ll = 0; ll<sz; ll++){
			filteredpixel = 0.0f;
			sum = 0.0f;
			for (ii = 0; ii<sz; ii++)
				for (jj = 0; jj<sz; jj++){
					temp = expf(-(ii - kk)*(ii - kk)*norm)*expf(-(ll - jj)*(ll - jj)*norm);
					filteredpixel += temp*data[ii*sz + jj];
					sum += temp;
				}
			filteredpixel /= sum;

			*MaxN = max(*MaxN, filteredpixel);
			*MinBG = min(*MinBG, filteredpixel);
		}
}

//***************************************************************************************************************************

__device__ void kernel_CentroidFitter(const int sz, const float *data, float *sx, float *sy,
	float *sx_std, float *sy_std){

	float tmpsx = 0.0f; float tmpsx_std = 0.0f;
	float tmpsy = 0.0f; float tmpsy_std = 0.0f;
	float tmpsum = 0.0f; float tmpsum_std = 0.0f;
	float min = 10000.0f;
	int ii, jj;
	float total = 0.0f;
	int center = (sz - 1) / 2;

	for (ii = 0; ii<sz; ii++)
		for (jj = 0; jj<sz; jj++)
			total += data[sz*jj + ii];

	float thrsh = total / (sz*sz);
	for (jj = 0; jj<sz; jj++)
		for (ii = 0; ii<sz; ii++){
			if (data[sz*jj + ii]>thrsh){
				tmpsx += data[sz*jj + ii] * ii;
				tmpsy += data[sz*jj + ii] * jj;
				tmpsum += data[sz*jj + ii];
			}
		}

	*sx = tmpsx / tmpsum;
	*sy = tmpsy / tmpsum;

	if (*sx> (center + 1)){
		*sx = center + 1;
	}
	else if (*sx<(center - 1)){
		*sx = center - 1;
	}


	if (*sy >(center + 1)){
		*sy = center + 1;
	}
	else if (*sy <(center - 1)){
		*sy = center - 1;
	}


	for (ii = 0; ii<sz; ii++)
		for (jj = 0; jj<sz; jj++) {
			if (data[sz*jj + ii]<min){
				min = data[sz*jj + ii];
			}
		}

	for (ii = 0; ii<sz; ii++)
		for (jj = 0; jj<sz; jj++) {
			if (data[sz*jj + ii]>thrsh){
				tmpsum_std += (data[sz*jj + ii] - min);
				tmpsx_std += (data[sz*jj + ii] - min)*(ii - *sx)*(ii - *sx);
				tmpsy_std += (data[sz*jj + ii] - min)*(jj - *sy)*(jj - *sy);
			}
		}
	tmpsx_std /= tmpsum_std;
	tmpsy_std /= tmpsum_std;
	*sx_std = tmpsx_std;
	*sy_std = tmpsy_std;
}

//*******************************************************************************************
// Global Calls
//*******************************************************************************************

extern "C"
__global__ void kernel_MLEFit(float *d_data, float PSFSigma, int sz, int iterations,
float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits){
	float M[NV_P*NV_P], Diag[NV_P], Minv[NV_P*NV_P];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BlockSize = blockDim.x;
	int ii, jj, kk, ll;
	float model, cf, df, data;
	float Div;
	float PSFy, PSFx;
	int NV = NV_P;
	float dudt[NV_P];
	float d2udt2[NV_P];
	float NR_Numerator[NV_P], NR_Denominator[NV_P];
	float theta[NV_P];
	float maxjump[] = { 1e0f, 1e0f, 1e2f, 2e0f };
	float gamma[] = { 1.0f, 1.0f, 0.5f, 1.0f };
	float Nmax;

	//Prevent read/write past end of array
	if ((bx*BlockSize + tx) >= Nfits) return;

	memset(M, 0, NV_P*NV_P*sizeof(float));
	memset(Minv, 0, NV_P*NV_P*sizeof(float));
	//load data
	float *s_data = d_data + (sz*sz*bx*BlockSize + sz*sz*tx);
	//initial values
	kernel_CenterofMass2D(sz, s_data, &theta[0], &theta[1]);
	kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &theta[3]);
	theta[2] = fmaxf(0.0f, (Nmax - theta[3]) * 2 * pi*PSFSigma*PSFSigma);

	for (kk = 0; kk<iterations; kk++) {//main iterative loop

		//initialize
		memset(NR_Numerator, 0, NV_P*sizeof(float));
		memset(NR_Denominator, 0, NV_P*sizeof(float));

		for (ii = 0; ii<sz; ii++)
			for (jj = 0; jj<sz; jj++) {
				PSFx = kernel_IntGauss1D(ii, theta[0], PSFSigma);
				PSFy = kernel_IntGauss1D(jj, theta[1], PSFSigma);

				model = theta[3] + theta[2] * PSFx*PSFy;
				data = s_data[sz*jj + ii];

				//calculating derivatives
				kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
				kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
				dudt[2] = PSFx*PSFy;
				d2udt2[2] = 0.0f;
				dudt[3] = 1.0f;
				d2udt2[3] = 0.0f;

				cf = 0.0f;
				df = 0.0f;
				if (model>10e-3f) cf = data / model - 1;
				if (model>10e-3f) df = data / (model * model);
				cf = min(cf, 10e5f);
				df = min(df, 10e5f);

				for (ll = 0; ll<NV; ll++){
					NR_Numerator[ll] += dudt[ll] * cf;
					NR_Denominator[ll] += d2udt2[ll] * cf - dudt[ll] * dudt[ll] * df;
				}
			}

		// The update
		if (kk<2)
			for (ll = 0; ll<NV; ll++)
				theta[ll] -= gamma[ll] * min(max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
		else
			for (ll = 0; ll<NV; ll++)
				theta[ll] -= min(max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

		// Any other constraints
		theta[2] = max(theta[2], 1.0f);
		theta[3] = max(theta[3], 0.01f);
	}

	// Calculating the CRLB and LogLikelihood
	Div = 0.0;
	for (ii = 0; ii<sz; ii++)
		for (jj = 0; jj<sz; jj++) {
			PSFx = kernel_IntGauss1D(ii, theta[0], PSFSigma);
			PSFy = kernel_IntGauss1D(jj, theta[1], PSFSigma);

			model = theta[3] + theta[2] * PSFx*PSFy;
			data = s_data[sz*jj + ii];

			//calculating derivatives
			kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
			kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
			dudt[2] = PSFx*PSFy;
			dudt[3] = 1.0f;

			//Building the Fisher Information Matrix
			for (kk = 0; kk<NV; kk++)
				for (ll = kk; ll<NV; ll++){
					M[kk*NV + ll] += dudt[ll] * dudt[kk] / model;
					M[ll*NV + kk] = M[kk*NV + ll];
				}

			//LogLikelyhood
			if (model>0){
				if (data>0){
					Div += data*logf(model) - model - data*logf(data) + data;
				}
				else {
					Div += -model;
				}
			}
		}

	// Matrix inverse (CRLB=F^-1) and output assignments
	kernel_MatInvN(M, Minv, Diag, NV);

	//write to global arrays
	for (kk = 0; kk<NV; kk++)
		d_Parameters[Nfits*kk + BlockSize*bx + tx] = theta[kk];
	for (kk = 0; kk<NV; kk++)
		d_CRLBs[Nfits*kk + BlockSize*bx + tx] = Diag[kk];
	d_LogLikelihood[BlockSize*bx + tx] = Div;

	return;
}

extern "C"
__global__ void kernel_MLEFit_sigma(float *d_data, float PSFSigma, int sz, int iterations,
float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits){

	float M[NV_PS*NV_PS], Diag[NV_PS], Minv[NV_PS*NV_PS];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BlockSize = blockDim.x;
	int ii, jj, kk, ll;
	float model, cf, df, data;
	float Div;
	float PSFy, PSFx;
	int NV = NV_PS;
	float dudt[NV_PS];
	float d2udt2[NV_PS];
	float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
	float theta[NV_PS];
	float maxjump[NV_PS] = { 1.0f, 1.0f, 200.0f, 10.0f, 0.1f};
	float gamma[NV_PS] = { 1.0f, 1.0f, 0.5f, 1.0f, 1.0f };
	float Nmax;
	float diff;
	float sums[NV_PS];

	//Prevent read/write past end of array
	if ((bx*BlockSize + tx) >= Nfits) return;

	memset(M, 0, NV_P*NV_P*sizeof(float));
	memset(Minv, 0, NV_P*NV_P*sizeof(float));
	memset(sums, 0, NV*sizeof(float));
	//load data
	float *s_data = d_data + (sz*sz*bx*BlockSize + sz*sz*tx);

	//initial values
	kernel_CenterofMass2D(sz, s_data, &theta[0], &theta[1]);
	kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &theta[3]);
	theta[2] = fmaxf(0.0f, (Nmax - theta[3]) * 2 * pi*PSFSigma*PSFSigma);
	theta[4] = PSFSigma;
	d2udt2[2] = 0.0f;
	dudt[3] = 1.0f;
	d2udt2[3] = 0.0f;

	for (kk = 0; kk<iterations; kk++) {//main iterative loop

		//initialize
		memset(NR_Numerator, 0, NV_P*sizeof(float));
		memset(NR_Denominator, 0, NV_P*sizeof(float));

		for (ii = 0; ii<sz; ii++){
			PSFy = kernel_IntGauss1D(jj, theta[1], theta[4]);
			for (jj = 0; jj<sz; jj++) {
				PSFx = kernel_IntGauss1D(ii, theta[0], theta[4]);

				model = theta[3] + theta[2] * PSFx * PSFy;
				data = s_data[sz*jj + ii];

				//calculating derivatives					
				kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            	kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            	kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
				dudt[2] = PSFx * PSFy;

				cf = 0.0f;
				df = 0.0f;
				if (model>10e-3f) df = data / (model * model);
				if (model>10e-3f) cf = data / model - 1;
				cf=min(cf, 10e4f);
            	df=min(df, 10e4f);

				for (ll = 0; ll<NV; ll++){
					NR_Numerator[ll] += dudt[ll] * cf;
					NR_Denominator[ll] += d2udt2[ll] * cf - dudt[ll] * dudt[ll] * df;
				}
			}
		}

		// The update
		for (ll = 0; ll<NV; ll++){
			if (kk<5)
				diff = gamma[ll] * min(max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
			else
				diff = min(max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
			theta[ll] -= diff;
			if(kk>iterations-10)
				sums[ll]+=abs(diff);
		}

		// Any other constraints
		theta[2]=max(theta[2], 1.0f);
        theta[3]=max(theta[3], 0.001f);
        theta[4]=max(theta[4], 0.5f);
        theta[4]=min(theta[4], PSFSigma/10.0f);
	}

	// Calculating the CRLB and LogLikelihood
	Div = 0.0f;
	for (ii = 0; ii<sz; ii++) for (jj = 0; jj<sz; jj++) {
		PSFx = kernel_IntGauss1D(ii, theta[0], PSFSigma);
		PSFy = kernel_IntGauss1D(jj, theta[1], PSFSigma);

		model = theta[3] + theta[2] * PSFx*PSFy;
		data = s_data[sz*jj + ii];

		//calculating derivatives
		kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
		kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], NULL);
		kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], NULL);
		dudt[2] = PSFx*PSFy;
		dudt[3] = 1.0f;

		//Building the Fisher Information Matrix
		for (kk = 0; kk<NV; kk++)for (ll = kk; ll<NV; ll++){
			M[kk*NV + ll] += dudt[ll] * dudt[kk] / model;
			M[ll*NV + kk] = M[kk*NV + ll];
		}

		//LogLikelyhood
		if (model>0)
			if (data>0)Div += data*logf(model) - model - data*logf(data) + data;
			else
				Div += -model;
	}

	// Matrix inverse (CRLB=F^-1) and output assigments
	//kernel_MatInvN(M, Minv, Diag, NV);

	//write to global arrays
	for (kk = 0; kk<NV; kk++)
		d_Parameters[Nfits*kk + BlockSize*bx + tx] = theta[kk];
	for (kk = 0; kk<NV; kk++)
		d_CRLBs[Nfits*kk + BlockSize*bx + tx] = sums[kk];
	d_LogLikelihood[BlockSize*bx + tx] = Div;

	return;
}

extern "C"
__global__ void kernel_MLEFit_sigmaxy(float *d_data, float PSFSigma, int sz, int iterations,
float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits){

	const int NV = NV_P2;
	float M[NV*NV], Diag[NV], Minv[NV*NV];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BlockSize = blockDim.x;
	int ii, jj, kk, ll;
	float model, cf, df, data;
	float Div;
	float PSFy, PSFx;
	float dudt[NV];
	float d2udt2[NV];
	float NR_Numerator[NV], NR_Denominator[NV];
	float theta[NV];
	float maxjump[NV] = { 1.0f, 1.0f, 200.0f, 10.0f, 0.1f, 0.1f };
	float g[NV] = { 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f };
	float Nmax;
	float diff;
	float sums[NV];

	//Prevent read/write past end of array
	if ((bx*BlockSize + tx) >= Nfits) return;

	memset(M, 0, NV*NV*sizeof(float));
	memset(Minv, 0, NV*NV*sizeof(float));
	memset(sums, 0, NV*sizeof(float));
	//load data
	float *s_data = d_data + (sz*sz*bx*BlockSize + sz*sz*tx);

	//initial values
	kernel_CenterofMass2D(sz, s_data, &theta[0], &theta[1]);
	kernel_GaussFMaxMin2D(sz, PSFSigma, s_data, &Nmax, &theta[3]);
	theta[2] = max(0.0f, (Nmax - theta[3]) * 2 * pi*PSFSigma*PSFSigma);
	theta[4] = PSFSigma;
	theta[5] = PSFSigma;
	d2udt2[2] = 0.0f;
	dudt[3] = 1.0f;
	d2udt2[3] = 0.0f;

	for (kk = 0; kk<iterations; kk++) {//main iterative loop

		//initialize
		memset(NR_Numerator, 0, NV*sizeof(float));
		memset(NR_Denominator, 0, NV*sizeof(float));

		for (jj = 0; jj<sz; jj++){
			PSFy = kernel_IntGauss1D(jj, theta[1], theta[5]);
			
			for (ii = 0; ii<sz; ii++) {
				PSFx = kernel_IntGauss1D(ii, theta[0], theta[4]);
				
				model = theta[3] + theta[2] * PSFx*PSFy;
				data = s_data[sz*jj + ii];

				//calculating derivatives
				kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
				kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], &d2udt2[4]);
				kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], &d2udt2[1]);
				kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], &d2udt2[5]);
				dudt[2] = PSFx*PSFy;

				cf = 0.0f;
				df = 0.0f;
				if (model>10e-3f) df = data / (model * model);
				if (model>10e-3f) cf = data / model - 1;
				df = min(df, 10e5f);
				cf = min(cf, 10e5f);

				for (ll = 0; ll<NV; ll++){
					NR_Numerator[ll] += dudt[ll] * cf;
					NR_Denominator[ll] += d2udt2[ll] * cf - dudt[ll] * dudt[ll] * df;
				}
			}
		}

		// The update
		for (ll = 0; ll<NV; ll++){
			diff = g[ll] * min(max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
			theta[ll] -= diff;
			if(kk>iterations-10)
				sums[ll]+=abs(diff);
		}

		// Any other constraints
		theta[2] = max(theta[2], 1.0f);
		theta[3] = max(theta[3], 0.001f);
		theta[4] = max(theta[4], PSFSigma / 20.0f);
		theta[5] = max(theta[5], PSFSigma / 20.0f);
	}

	// Calculating the CRLB and LogLikelihood
	Div = 0.0f;
	dudt[3] = 1.0f;
	for (jj = 0; jj<sz; jj++){
		PSFy = kernel_IntGauss1D(jj, theta[1], theta[5]);
		for (ii = 0; ii<sz; ii++)  {
			PSFx = kernel_IntGauss1D(ii, theta[0], theta[4]);
			
			model = theta[3] + theta[2] * PSFx*PSFy;
			data = s_data[sz*jj + ii];
	
			//calculating derivatives
			kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
			kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
			kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
			kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);
			dudt[2] = PSFx*PSFy;
	
			//Building the Fisher Information Matrix
			for (kk = 0; kk<NV; kk++)
				for (ll = kk; ll<NV; ll++){
				M[kk*NV + ll] += dudt[ll] * dudt[kk] / model;
				M[ll*NV + kk] = M[kk*NV + ll];
			}
	
			//LogLikelyhood
			if (model>0)
				if (data>0)Div += data*logf(model) - model - data*logf(data) + data;
				else
					Div += -model;
		}
	}

	// Matrix inverse (CRLB=F^-1) and output assigments
 	//kernel_MatInvN(M, Minv, Diag, NV);

	//write to global arrays
	for (kk = 0; kk<NV; kk++)
		d_Parameters[Nfits*kk + BlockSize*bx + tx] = theta[kk];
	for (kk = 0; kk<NV; kk++)
		d_CRLBs[Nfits*kk + BlockSize*bx + tx] = sums[kk];
	d_LogLikelihood[BlockSize*bx + tx] = Div;
	return;
}
