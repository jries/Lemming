package org.lemming.tools;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.lemming.pipeline.Kernel;

import javolution.util.FastTable;

class MLE implements Callable<Map<String,float[]>>{
	
	private static final double sq2pi = FastMath.sqrt(2*FastMath.PI);
	private static final int NV_P=4;			//!< number of fitting parameters for MLEfit (x,y,bg,I)
	private static final int NV_P2=6;			//!< number of fitting parameters for MLEFit_sigmaxy (x,y,bg,I,Sx,Sy)
	private final List<Kernel> kList;
	private final int nKernels;
	private final int sz;
	private static final double PSFSigma = 1.3f;
	private static final int iterations = 200;
	private static final double sharedMemPerBlock = 262144;
	
	public MLE(List<Kernel> kernelList, int sz, int nKernels){
		this.sz = sz;
		this.nKernels = nKernels;
		this.kList = kernelList;
	}
	
	@Override
	public Map<String, float[]> call() throws Exception {
		float[] Ival = new float[sz*sz*nKernels];
		int sliceIndex = 0;
	    for(int k=0;k<nKernels;k++){
	        Kernel kernel = kList.get(k);
	        float[] values = kernel.getValues();

	        for(int l=0;l<values.length;l++){
	            int index = sliceIndex + l;
	            Ival[index] = values[l];
	        }
	        sliceIndex += values.length;
	    }
	    
	    /* C CODE */
	    int BlockSize = (int) FastMath.floor(sharedMemPerBlock/4/sz/sz);
	    BlockSize = FastMath.max(9, BlockSize);
	    BlockSize = FastMath.min(288, BlockSize);
	    int gridSize = (int) FastMath.ceil((double)nKernels / BlockSize);
	    FastTable<Double> d_Parameters = new FastTable<>();
	    for (int i=0; i<nKernels*6;i++)
	    	d_Parameters.add(0d);
	    FastTable<Double>d_CRLBs = new FastTable<>();
	    for (int i=0; i<nKernels*6;i++)
	    	d_CRLBs.add(0d);
	    FastTable<Double>d_LogLikelihood = new FastTable<>();
	    for (int i=0; i<nKernels;i++)
	    	d_LogLikelihood.add(0d);
	    for (int bx=0;bx<gridSize;bx++)
	        for(int tx=0;tx<BlockSize;tx++)
	            kernel_MLEFit_sigmaxy(Ival, PSFSigma, sz, iterations,
	                  d_Parameters, d_CRLBs, d_LogLikelihood,
	                  nKernels, tx, bx, BlockSize);
	    
	    HashMap<String, float[]> result = new HashMap<>();
	    float hostParameters[] = new float[d_Parameters.size()];
	    for (int i=0; i<d_Parameters.size();i++)
	    	hostParameters[i]=d_Parameters.get(i).floatValue();
	    float hostCRLBs[] = new float[d_Parameters.size()];
	    for (int i=0; i<d_Parameters.size();i++)
	    	hostCRLBs[i]=d_CRLBs.get(i).floatValue();
	    result.put("Parameters", hostParameters);
	    result.put("CRLBs", hostCRLBs);
		return result;
	}	
	
	//*******************************************************************************************
	// Private Calls
	//*******************************************************************************************
	private void kernel_MatInvN(double[] M, double[] Minv, double[] DiagMinv, int sz) {

		int ii, jj, kk, num, b;
		double tmp1 = 0;
		double yy[]=new double[25];

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

		if (DiagMinv!=null)
			for (ii = 0; ii < sz; ii++)
				DiagMinv[ii] = Minv[ii * sz + ii];

	}

	//*******************************************************************************************
	private double kernel_IntGauss1D(final int ii, final double theta, final double sigma) {
		final double norm = 0.5f / sigma / sigma;
		return 0.5f
			* (Erf.erf((ii - theta + 0.5f) * FastMath.sqrt(norm))
			- Erf.erf((ii - theta - 0.5f) * FastMath.sqrt(norm)));
	}

	//*******************************************************************************************
	 double kernel_alpha(final double z, final double Ax, final double Bx, final double d) {
		double q = z / d;
		return 1.0f + (q*q) + Ax * (q*q*q) + Bx * (q*q*q*q);
	}

	//*******************************************************************************************
	 double kernel_dalphadz(final double z, final double Ax, final double Bx, final double d) {
		return (2.0f * z / (d*d) + 3.0f * Ax * (z*z) / (d*d*d)
			+ 4.0f * Bx * (z*z*z) / (d*d*d*d));
	}

	//*******************************************************************************************
	 double kernel_d2alphadz2(final double z, final double Ax, final double Bx, final double d) {
		return (2.0f / (d*d) + 6.0f * Ax * z / (d*d*d)
			+ 12.0f * Bx * (z*z) / (d*d*d*d));
	}

	//*******************************************************************************************
	private void kernel_DerivativeIntGauss1D(final int ii, final double theta,
											 final double sigma, final double theta2, final double PSFy, MutableDouble dudt,
											 MutableDouble d2udt2) {

		double a, b;
		a = FastMath.exp(-0.5f * ((ii + 0.5f - theta) / sigma)*((ii + 0.5f - theta) / sigma));
		b = FastMath.exp(-0.5f * ((ii - 0.5f - theta) / sigma)*((ii - 0.5f - theta) / sigma));

		dudt.setValue(-theta2 / sq2pi / sigma * (a - b) * PSFy);

		if (d2udt2!=null)
			d2udt2.setValue(-theta2 / sq2pi / (sigma*sigma*sigma)
			* ((ii + 0.5f - theta) * a - (ii - 0.5f - theta) * b) * PSFy);
	}

	//*******************************************************************************************
	private void kernel_DerivativeIntGauss1DSigma(final int ii, final double x,
												  final double Sx, final double N, final double PSFy, MutableDouble dudt,
												  MutableDouble d2udt2, double dudt0) {

		double ax, bx;

		ax = FastMath.exp(-0.5f * ((ii + 0.5f - x) / Sx)*((ii + 0.5f - x) / Sx));
		bx = FastMath.exp(-0.5f * ((ii - 0.5f - x) / Sx)*((ii - 0.5f - x) / Sx));
		dudt.setValue(-N / sq2pi / Sx / Sx
			* (ax * (ii - x + 0.5f) - bx * (ii - x - 0.5f)) * PSFy);

		if (d2udt2!=null)
			d2udt2.setValue(-2.0f / Sx * dudt0
			- N / sq2pi / (Sx*Sx*Sx*Sx*Sx)
			* (ax * ((ii - x + 0.5f)*(ii - x + 0.5f)*(ii - x + 0.5f))
			- bx * ((ii - x - 0.5f)*(ii - x - 0.5f)*(ii - x - 0.5f))) * PSFy);
	}

	//*******************************************************************************************
	 void kernel_DerivativeIntGauss2DSigma(final int ii, final int jj,
		final double x, final double y, final double S, final double N,
		final double PSFx, final double PSFy, MutableDouble dudt, MutableDouble d2udt2, double dudt0) {

		MutableDouble dSx = new MutableDouble(), dSy = new MutableDouble(), ddSx = new MutableDouble(), ddSy = new MutableDouble();

		kernel_DerivativeIntGauss1DSigma(ii, x, S, N, PSFy, dSx, ddSx, dudt0);
		kernel_DerivativeIntGauss1DSigma(jj, y, S, N, PSFx, dSy, ddSy, dudt0);

		dudt.setValue(dSx.getValue() + dSy.getValue());
		if (d2udt2!=null)
			d2udt2.setValue(ddSx.getValue() + ddSy.getValue());
	}

	//*******************************************************************************************
	private void kernel_CenterofMass2D(final int sz, final float[] d_data, int index, MutableDouble x, MutableDouble y) {
		double tmpx = 0.0f;
		double tmpy = 0.0f;
		double tmpsum = 0.0f;
		int ii, jj;
		for (jj = 0; jj<sz; jj++)
			for (ii = 0; ii<sz; ii++){
				tmpx += d_data[index + (sz*jj + ii)] * ii;
				tmpy += d_data[index + (sz*jj + ii)] * jj;
				tmpsum += d_data[index + (sz*jj + ii)];
			}
		x.setValue(tmpx / tmpsum);
		y.setValue(tmpy / tmpsum);
	}

	// *******************************************************************************************
	private void kernel_GaussFMaxMin2D(final int sz, final double sigma, float[] data, int index, MutableDouble MaxN, MutableDouble MinBG) {

		int ii, jj, kk, ll;
		double filteredpixel, sum;
		double temp;
		MaxN.setValue(0);
		MinBG.setValue(10e10); // big

		double norm = 0.5f / sigma / sigma;
		// loop over all pixels
		for (kk = 0; kk < sz; kk++)
			for (ll = 0; ll < sz; ll++) {
				filteredpixel = 0.0f;
				sum = 0.0f;
				for (jj = 0; jj < sz; jj++)
					for (ii = 0; ii < sz; ii++) {
						temp = FastMath.exp(-(ii - kk) * (ii - kk) * norm) * FastMath.exp(-(ll - jj) * (ll - jj) * norm);
						filteredpixel += temp * data[index + (sz * jj + ii)];
						sum += temp;
					}
				filteredpixel /= sum;

				MaxN.setValue(FastMath.max(MaxN.getValue(), filteredpixel));
				MinBG.setValue(FastMath.min(MinBG.getValue(), filteredpixel));
			}
	}

	//***************************************************************************************************************************
	private void kernel_CentroidFitter(final int sz, final float[] data, final int index, MutableDouble sx, MutableDouble sy,
									   MutableDouble sx_std, MutableDouble sy_std){

		double tmpsx = 0.0f; double tmpsx_std = 0.0f;
		double tmpsy = 0.0f; double tmpsy_std = 0.0f;
		double tmpsum = 0.0f; double tmpsum_std = 0.0f;
		double min = 10000.0f;
		int ii, jj;
		double total = 0.0f;
		int center = (sz - 1) / 2;

		for (ii = 0; ii<sz; ii++)
			for (jj = 0; jj<sz; jj++)
				total += data[sz*jj + ii];

		double thrsh = total / (sz*sz);
		for (jj = 0; jj<sz; jj++)
			for (ii = 0; ii<sz; ii++){
				if (data[sz*jj + ii]>thrsh){
					tmpsx += data[index + (sz*jj + ii)] * ii;
					tmpsy += data[index + (sz*jj + ii)] * jj;
					tmpsum += data[index + (sz*jj + ii)];
				}
			}

		sx.setValue(tmpsx / tmpsum);
		sy.setValue(tmpsy / tmpsum);

		if (sx.getValue()> (center + 1)){
			sx.setValue(center + 1);
		}
		else if (sx.getValue()<(center - 1)){
			sx.setValue(center - 1);
		}


		if (sy.getValue() >(center + 1)){
			sy.setValue(center + 1);
		}
		else if (sy.getValue() <(center - 1)){
			sy.setValue(center - 1);
		}


		for (jj = 0; jj<sz; jj++)
			for (ii = 0; ii<sz; ii++) {
				if (data[index + (sz*jj + ii)]<min){
					min = data[index + (sz*jj + ii)];
				}
			}

		for (jj = 0; jj<sz; jj++)
			for (ii = 0; ii<sz; ii++) {
				if (data[sz*jj + ii]>thrsh){
					tmpsum_std += (data[index + (sz*jj + ii)] - min);
					tmpsx_std += (data[index + (sz*jj + ii)] - min)*(ii - sx.getValue())*(ii - sx.getValue());
					tmpsy_std += (data[index + (sz*jj + ii)] - min)*(jj - sy.getValue())*(jj - sy.getValue());
				}
			}
		tmpsx_std /= tmpsum_std;
		tmpsy_std /= tmpsum_std;
		sx_std.setValue(tmpsx_std);
		sy_std.setValue(tmpsy_std);
	}
	 
	//*******************************************************************************************
	// Public Calls
	//*******************************************************************************************
	public void kernel_MLEFit(float[] d_data, double PSFSigma, int sz, int iterations,
			double[]d_Parameters, double[]d_CRLBs, double[]d_LogLikelihood, int Nfits, int threadIdx, int blockIdx, int blockDim){
		double[] M = new double[NV_P*NV_P]; 
		double[] Diag = new double[NV_P]; 
		double[] Minv= new double[NV_P*NV_P];
		int ii, jj, kk, ll;
		double model, cf, df, data;
		double Div;
		double PSFy, PSFx;
		int NV = NV_P;
		MutableDouble[] dudt = new MutableDouble[NV_P];
		MutableDouble[] d2udt2 = new MutableDouble[NV_P];
		double[] NR_Numerator = new double[NV_P];
		double[] NR_Denominator = new double[NV_P];
		MutableDouble[] theta = new MutableDouble[NV_P];
		double maxjump[] = { 1e0f, 1e0f, 1e2f, 2e0f };
		double gamma[] = { 1.0f, 1.0f, 0.5f, 1.0f };
		MutableDouble Nmax = new MutableDouble();

		//Prevent read/write past end of array
		if ((blockIdx * blockDim + threadIdx) >= Nfits) return;

		Arrays.fill(M, 0);
		Arrays.fill(Minv, 0);
		//load data
		int index = sz*sz* blockIdx * blockDim + sz*sz* threadIdx;
		//initial values
		
		kernel_CenterofMass2D(sz, d_data, index, theta[0], theta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, d_data, index, Nmax, theta[3]);
		theta[2].setValue(FastMath.max(0.0f, (Nmax.getValue() - theta[3].getValue()) * 2 * FastMath.PI * PSFSigma * PSFSigma));

		for (kk = 0; kk<iterations; kk++) {//main iterative loop

			//initialize
			Arrays.fill(NR_Numerator, 0);
			Arrays.fill(NR_Denominator, 0);

			for (ii = 0; ii<sz; ii++)
				for (jj = 0; jj<sz; jj++) {
					PSFx = kernel_IntGauss1D(ii, theta[0].getValue(), PSFSigma);
					PSFy = kernel_IntGauss1D(jj, theta[1].getValue(), PSFSigma);

					model = theta[3].getValue() + theta[2].getValue() * PSFx*PSFy;
					data = d_data[index + (sz*jj + ii)];

					//calculating derivatives
					kernel_DerivativeIntGauss1D(ii, theta[0].getValue(), PSFSigma, theta[2].getValue(), PSFy, dudt[0], d2udt2[0]);
					kernel_DerivativeIntGauss1D(jj, theta[1].getValue(), PSFSigma, theta[2].getValue(), PSFx, dudt[1], d2udt2[1]);
					dudt[2].setValue(PSFx*PSFy);
					d2udt2[2].setValue(0.0f);
					dudt[3].setValue(1.0f);
					d2udt2[3].setValue(0.0f);

					cf = 0.0f;
					df = 0.0f;
					if (model>10e-3f) cf = data / model - 1;
					if (model>10e-3f) df = data / (model * model);
					cf = FastMath.min(cf, 10e5f);
					df = FastMath.min(df, 10e5f);

					for (ll = 0; ll<NV; ll++){
						NR_Numerator[ll] += dudt[ll].getValue() * cf;
						NR_Denominator[ll] += d2udt2[ll].getValue() * cf - dudt[ll].getValue() * dudt[ll].getValue() * df;
					}
				}

			// The update
			if (kk<2)
				for (ll = 0; ll<NV; ll++)
					theta[ll].subtract(gamma[ll] * FastMath.min(FastMath.max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]));
			else
				for (ll = 0; ll<NV; ll++)
					theta[ll].subtract(FastMath.min(FastMath.max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]));

			// Any other constraints
			theta[2].setValue(FastMath.max(theta[2].getValue(), 1.0f));
			theta[3].setValue(FastMath.max(theta[3].getValue(), 0.01f));
		}

		// Calculating the CRLB and LogLikelihood
		Div = 0.0f;
		for (ii = 0; ii<sz; ii++)
			for (jj = 0; jj<sz; jj++) {
				PSFx = kernel_IntGauss1D(ii, theta[0].getValue(), PSFSigma);
				PSFy = kernel_IntGauss1D(jj, theta[1].getValue(), PSFSigma);

				model = theta[3].getValue() + theta[2].getValue() * PSFx*PSFy;
				data = d_data[index + (sz*jj + ii)];

				//calculating derivatives
				kernel_DerivativeIntGauss1D(ii, theta[0].getValue(), PSFSigma, theta[2].getValue(), PSFy, dudt[0], null);
				kernel_DerivativeIntGauss1D(jj, theta[1].getValue(), PSFSigma, theta[2].getValue(), PSFx, dudt[1], null);
				dudt[2].setValue(PSFx*PSFy);
				dudt[3].setValue(1.0f);

				//Building the Fisher Information Matrix
				for (kk = 0; kk<NV; kk++)
					for (ll = kk; ll<NV; ll++){
						M[kk*NV + ll] += dudt[ll].getValue() * dudt[kk].getValue() / model;
						M[ll*NV + kk] = M[kk*NV + ll];
					}

				//LogLikelyhood
				if (model>0){
					if (data>0){
						Div += data*FastMath.log(model) - model - data*FastMath.log(data) + data;
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
			d_Parameters[Nfits*kk + blockDim * blockIdx + threadIdx] = theta[kk].getValue();
		for (kk = 0; kk<NV; kk++)
			d_CRLBs[Nfits*kk + blockDim * blockIdx + threadIdx] = Diag[kk];
		d_LogLikelihood[blockDim * blockIdx + threadIdx] = Div;

	}
	
	private void kernel_MLEFit_sigmaxy(float[] d_data, double PSFSigma, int sz, int iterations,
									   FastTable<Double> d_Parameters, FastTable<Double> d_CRLBs, FastTable<Double> d_LogLikelihood, int Nfits, int threadIdx, int blockIdx, int blockDim){

		final int NV = NV_P2;
		double[] M = new double[NV*NV]; 
		//double[] Diag = new double[NV]; 
		double[] Minv= new double[NV*NV];
		int ii, jj, kk, ll;
		double model, cf, df, data;
		//double Div;
		double PSFy, PSFx;
		double maxjump[] = { 1.0f, 1.0f, 200.0f, 10.0f, 0.1f, 0.1f };
		double g[] = { 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f };
		MutableDouble[] dudt = new MutableDouble[NV];
		for (int i=0;i<NV;i++)
			dudt[i]=new MutableDouble();
		MutableDouble[] d2udt2 = new MutableDouble[NV];
		for (int i=0;i<NV;i++)
			d2udt2[i]=new MutableDouble();
		double[] NR_Numerator = new double[NV];
		double[] NR_Denominator = new double[NV];
		MutableDouble[] theta = new MutableDouble[NV];
		for (int i=0;i<NV;i++)
			theta[i]=new MutableDouble();
		MutableDouble Nmax = new MutableDouble();
		double diff;
		double sums[]=new double[NV];

		//Prevent read/write past end of array
		if ((blockIdx * blockDim + threadIdx) >= Nfits) return;

		Arrays.fill(M, 0);
		Arrays.fill(Minv, 0);
		//load data
		int index = (sz*sz* blockIdx * blockDim + sz*sz* threadIdx);

		//initial values
		kernel_CenterofMass2D(sz, d_data, index, theta[0], theta[1]);
		kernel_GaussFMaxMin2D(sz, PSFSigma, d_data, index, Nmax, theta[3]);
		theta[2].setValue(FastMath.max(0.0f, (Nmax.getValue() - theta[3].getValue()) * 3 * FastMath.PI * PSFSigma * PSFSigma));
		theta[4].setValue(PSFSigma);
		theta[5].setValue(PSFSigma);
		d2udt2[2].setValue(0.0f);
		dudt[3].setValue(1.0f);
		d2udt2[3].setValue(0.0f);

		for (kk = 0; kk<iterations; kk++) {//main iterative loop

			//initialize
			Arrays.fill(NR_Numerator, 0);
			Arrays.fill(NR_Denominator, 0);

			for (jj = 0; jj<sz; jj++){
				PSFy = kernel_IntGauss1D(jj, theta[1].getValue(), theta[5].getValue());
				
				for (ii = 0; ii<sz; ii++) {
					PSFx = kernel_IntGauss1D(ii, theta[0].getValue(), theta[4].getValue());
					
					model = theta[3].getValue() + theta[2].getValue() * PSFx*PSFy;
					data = d_data[index + (sz*jj + ii)];

					//calculating derivatives
					kernel_DerivativeIntGauss1D(jj, theta[1].getValue(), theta[5].getValue(), theta[2].getValue(), PSFx, dudt[1], d2udt2[1]);
					kernel_DerivativeIntGauss1D(ii, theta[0].getValue(), theta[4].getValue(), theta[2].getValue(), PSFy, dudt[0], d2udt2[0]);
					kernel_DerivativeIntGauss1DSigma(jj, theta[1].getValue(), theta[5].getValue(), theta[2].getValue(), PSFx, dudt[5], d2udt2[5], dudt[1].getValue());
					kernel_DerivativeIntGauss1DSigma(ii, theta[0].getValue(), theta[4].getValue(), theta[2].getValue(), PSFy, dudt[4], d2udt2[4], dudt[0].getValue());
					dudt[2].setValue(PSFx*PSFy);	

					cf = 0.0f;
					df = 0.0f;
					if (model>10e-3f) df = data / (model * model);
					if (model>10e-3f) cf = data / model - 1;
					df = FastMath.min(df, 10e5f);
					cf = FastMath.min(cf, 10e5f);

					for (ll = 0; ll<NV; ll++){
						NR_Numerator[ll] += dudt[ll].getValue() * cf;
						NR_Denominator[ll] += d2udt2[ll].getValue() * cf - dudt[ll].getValue() * dudt[ll].getValue() * df;
					}
				}
			}

			// The update
			for (ll = 0; ll<NV; ll++){
				diff = g[ll] * FastMath.min(FastMath.max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
				theta[ll].subtract(diff);
				if(kk>iterations-10)
					sums[ll]+=FastMath.abs(diff);
			}

			// Any other constraints
			theta[2].setValue(FastMath.max(theta[2].getValue(), 1.0f));
			theta[3].setValue(FastMath.max(theta[3].getValue(), 0.001f));
			theta[4].setValue(FastMath.max(theta[4].getValue(), PSFSigma / 20.0f));
			theta[5].setValue(FastMath.max(theta[5].getValue(), PSFSigma / 20.0f));
		}
		
		/*for (ll = 0; ll<NV; ll++){
			sums[ll]/=10;
		}*/

		// Calculating the CRLB and LogLikelihood
		/*
		Div = 0.0f;
		dudt[3].setValue(1.0f);
		for (jj = 0; jj<sz; jj++){
			PSFy = kernel_IntGauss1D(jj, theta[1].getValue(), theta[5].getValue());			
			for (ii = 0; ii<sz; ii++)  {
				PSFx = kernel_IntGauss1D(ii, theta[0].getValue(), theta[4].getValue());
	
				model = theta[3].getValue() + theta[2].getValue() * PSFx*PSFy;
				data = d_data[index + (sz*jj + ii)];
	
				//calculating derivatives
				kernel_DerivativeIntGauss1D(jj, theta[1].getValue(), theta[5].getValue(), theta[2].getValue(), PSFx, dudt[1], null);
				kernel_DerivativeIntGauss1DSigma(jj, theta[1].getValue(), theta[5].getValue(), theta[2].getValue(), PSFx, dudt[5], null, dudt[0].getValue());
				kernel_DerivativeIntGauss1D(ii, theta[0].getValue(), theta[4].getValue(), theta[2].getValue(), PSFy, dudt[0], null);
				kernel_DerivativeIntGauss1DSigma(ii, theta[0].getValue(), theta[4].getValue(), theta[2].getValue(), PSFy, dudt[4], null, dudt[0].getValue());
				dudt[2].setValue(PSFx*PSFy);
			

				//Building the Fisher Information Matrix
				for (kk = 0; kk<NV; kk++)
					for (ll = kk; ll<NV; ll++){
					M[kk*NV + ll] += dudt[ll].getValue() * dudt[kk].getValue() / model;
					M[ll*NV + kk] = M[kk*NV + ll];
				}

				//LogLikelyhood
				if (model>0)
					if (data>0)Div += data*FastMath.log(model) - model - data*FastMath.log(data) + data;
					else
						Div += -model;
			}
		}

		// Matrix inverse (CRLB=F^-1) and output assigments
	 	kernel_MatInvN(M, Minv, Diag, NV);
		 */
		//write to global arrays
		for (kk = 0; kk<NV; kk++)
			d_Parameters.set(Nfits*kk + blockDim * blockIdx + threadIdx, theta[kk].getValue());
		//for (kk = 0; kk<NV; kk++)
		//	d_CRLBs.set(Nfits*kk + BlockSize*bx + tx, Diag[kk]);
		//d_LogLikelihood.set(BlockSize*bx + tx, Div);
		for (kk = 0; kk<NV; kk++)
			d_CRLBs.set(Nfits*kk + blockDim * blockIdx + threadIdx, sums[kk]);
	}
	
	void kernel_MLEFit_z(
		float[] d_data, double PSFSigma_x, double Ax, double Ay, double Bx, double By, double gamma, double d, double PSFSigma_y, 
	    int sz, int iterations, FastTable<Double> d_Parameters, FastTable<Double> d_CRLBs,
		FastTable<Double> d_LogLikelihood, int Nfits, int threadIdx, int blockIdx, int blockDim) {

		final int NV = NV_P2;
		double[] M = new double[NV * NV];
		// double[] Diag = new double[NV];
		double[] Minv = new double[NV * NV];
		int ii, jj, kk, ll;
		double model, cf, df, data;
		// double Div;
		double PSFy, PSFx;
		double maxjump[] = { 1.0f, 1.0f, 200.0f, 10.0f, 0.1f, 0.1f };
		double g[] = { 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f };
		MutableDouble[] dudt = new MutableDouble[NV];
		for (int i = 0; i < NV; i++)
			dudt[i] = new MutableDouble();
		MutableDouble[] d2udt2 = new MutableDouble[NV];
		for (int i = 0; i < NV; i++)
			d2udt2[i] = new MutableDouble();
		double[] NR_Numerator = new double[NV];
		double[] NR_Denominator = new double[NV];
		MutableDouble[] theta = new MutableDouble[NV];
		for (int i = 0; i < NV; i++)
			theta[i] = new MutableDouble();
		MutableDouble Nmax = new MutableDouble();
		double diff;
		double sums[] = new double[NV];
		MutableDouble sigmay_sqrd = new MutableDouble();
		MutableDouble sigmax_sqrd = new MutableDouble();

		// Prevent read/write past end of array
		if ((blockIdx * blockDim + threadIdx) >= Nfits) return;

		Arrays.fill(M, 0);
		Arrays.fill(Minv, 0);
		// load data
		int index = (sz * sz * blockIdx * blockDim + sz * sz * threadIdx);

		// initial values
		//kernel_CenterofMass2D(sz, d_data, index, theta[0], theta[1]);
		kernel_CentroidFitter(sz, d_data, index, theta[0], theta[1], sigmax_sqrd, sigmay_sqrd);
		kernel_GaussFMaxMin2D(sz, PSFSigma, d_data, index, Nmax, theta[3]);
		theta[2].setValue(FastMath.max(0.0f, (Nmax.getValue() - theta[3].getValue()) * 3 * FastMath.PI * PSFSigma * PSFSigma));
		theta[4].setValue(d*d*(sigmay_sqrd.getValue()- sigmax_sqrd.getValue())/(4*gamma*PSFSigma_x*PSFSigma_y));
		d2udt2[2].setValue(0.0f);
		dudt[3].setValue(1.0f);
		d2udt2[3].setValue(0.0f);

		for (kk = 0; kk < iterations; kk++) {// main iterative loop

			// initialize
			Arrays.fill(NR_Numerator, 0);
			Arrays.fill(NR_Denominator, 0);

			for (jj = 0; jj < sz; jj++) {
				PSFy = kernel_IntGauss1D(jj, theta[1].getValue(), theta[5].getValue());

				for (ii = 0; ii < sz; ii++) {
					PSFx = kernel_IntGauss1D(ii, theta[0].getValue(), theta[4].getValue());

					model = theta[3].getValue() + theta[2].getValue() * PSFx * PSFy;
					data = d_data[index + (sz * jj + ii)];

					// calculating derivatives
					kernel_DerivativeIntGauss1D(jj, theta[1].getValue(), theta[5].getValue(), theta[2].getValue(), PSFx, dudt[1], d2udt2[1]);
					kernel_DerivativeIntGauss1D(ii, theta[0].getValue(), theta[4].getValue(), theta[2].getValue(), PSFy, dudt[0], d2udt2[0]);
					kernel_DerivativeIntGauss1DSigma(
						jj, theta[1].getValue(), theta[5].getValue(), theta[2].getValue(), PSFx, dudt[5], d2udt2[5], dudt[1].getValue());
					kernel_DerivativeIntGauss1DSigma(
						ii, theta[0].getValue(), theta[4].getValue(), theta[2].getValue(), PSFy, dudt[4], d2udt2[4], dudt[0].getValue());
					dudt[2].setValue(PSFx * PSFy);

					cf = 0.0f;
					df = 0.0f;
					if (model > 10e-3f) df = data / (model * model);
					if (model > 10e-3f) cf = data / model - 1;
					df = FastMath.min(df, 10e5f);
					cf = FastMath.min(cf, 10e5f);

					for (ll = 0; ll < NV; ll++) {
						NR_Numerator[ll] += dudt[ll].getValue() * cf;
						NR_Denominator[ll] += d2udt2[ll].getValue() * cf - dudt[ll].getValue() * dudt[ll].getValue() * df;
					}
				}
			}

			// The update
			for (ll = 0; ll < NV; ll++) {
				diff = g[ll] * FastMath.min(FastMath.max(NR_Numerator[ll] / NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
				theta[ll].subtract(diff);
				if (kk > iterations - 10) sums[ll] += FastMath.abs(diff);
			}

			// Any other constraints
			theta[2].setValue(FastMath.max(theta[2].getValue(), 1.0f));
			theta[3].setValue(FastMath.max(theta[3].getValue(), 0.001f));
			theta[4].setValue(FastMath.max(theta[4].getValue(), PSFSigma / 20.0f));
			theta[5].setValue(FastMath.max(theta[5].getValue(), PSFSigma / 20.0f));
		}
		// write to global arrays
		for (kk = 0; kk < NV; kk++)
			d_Parameters.set(Nfits * kk + blockDim * blockIdx + threadIdx, theta[kk].getValue());
		// for (kk = 0; kk<NV; kk++)
		// d_CRLBs.set(Nfits*kk + BlockSize*bx + tx, Diag[kk]);
		// d_LogLikelihood.set(BlockSize*bx + tx, Div);
		for (kk = 0; kk < NV; kk++)
			d_CRLBs.set(Nfits * kk + blockDim * blockIdx + threadIdx, sums[kk]);
	}

}
