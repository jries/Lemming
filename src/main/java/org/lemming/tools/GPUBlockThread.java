package org.lemming.tools;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import org.lemming.pipeline.Kernel;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;

public class GPUBlockThread implements Callable<Map<String,float[]>> {
	
	private int sz;
	private int sz2;
	private int nKernels;
	private CUdevice device;
	private List<Kernel> kList;
	private int PARAMETER_LENGTH;
	private String functionName;
	private static int count = 0;
	
	private static final float PSFSigma = 1.3f;
	private static final int iterations = 75;
	private static String ptxFileName = "resources/CudaFit.ptx";
	private static float sharedMemPerBlock = 262144;

	public GPUBlockThread(CUdevice device, List<Kernel> kernelList, int sz, int nKernels, int numParameters, String functionName) {
		this.sz = sz;
		this.sz2 = sz*sz;
		this.device = device;
		this.nKernels = nKernels;
		this.kList = kernelList;
		PARAMETER_LENGTH = numParameters;
		this.functionName = functionName;
	}
	
	@Override
    public Map<String, float[]> call() {
		int BlockSize = (int) Math.floor(sharedMemPerBlock/4/sz/sz);
		float[] Ival = new float[sz2*nKernels];
		for(int k=0;k<nKernels;k++){
			int sliceIndex = k*sz2;
			float[] values = kList.get(k).getValues();
			for (int l=0; l<sz2;l++)
				Ival[sliceIndex+l]=values[l];
		}
		return process(Ival, nKernels, BlockSize);
	}
	
	private Map<String,float[]> process(float data[], int Nfits, int blockSize){
    	long start = System.currentTimeMillis();
    	//put as many images as fit into a block
    	int BlockSize = Math.max(9, blockSize);
    	BlockSize = Math.min(288, BlockSize);
    	//int Nfits = BlockSize * (int) Math.ceil( (float) dims[2]/BlockSize);
    	int size = sz2*Nfits;
    	
    	CUcontext context = new CUcontext();
    	checkResult(cuCtxCreate(context, 0, device));
		//JCudaDriver.cuProfilerStart();
    	
    	// Load the ptx file.
        CUmodule module = new CUmodule();
        checkResult(cuModuleLoad(module, ptxFileName));
        // Obtain a function pointer to the needed function.
        CUfunction function = new CUfunction();
        checkResult(cuModuleGetFunction(function, module, functionName));
    	    	
    	// Allocate the device input data, and copy the host input data to the device
    	Pointer d_data = new Pointer();
    	checkResult(cudaMalloc(d_data, size * Sizeof.FLOAT));
    	checkResult(cudaMemcpy(d_data, Pointer.to(data), size * Sizeof.FLOAT, cudaMemcpyHostToDevice));
        
        // Allocate device output memory
    	Pointer d_Parameters = new Pointer();
    	checkResult(cudaMalloc(d_Parameters, PARAMETER_LENGTH * Nfits * Sizeof.FLOAT));
        Pointer d_CRLBs = new Pointer();
        checkResult(cudaMalloc(d_CRLBs, PARAMETER_LENGTH * Nfits * Sizeof.FLOAT));
        Pointer d_LogLikelihood = new Pointer();
        checkResult(cudaMalloc(d_LogLikelihood, Nfits * Sizeof.FLOAT));
        
        // Set up the kernel parameters: A pointer to an array
        // __global__ void kernel_MLEFit(float *d_data, float PSFSigma, int sz, int iterations, float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood, int Nfits)
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(d_data),
            Pointer.to(new float[]{PSFSigma}),
            Pointer.to(new int[]{sz}),
            Pointer.to(new int[]{iterations}),
            Pointer.to(d_Parameters),
            Pointer.to(d_CRLBs),
            Pointer.to(d_LogLikelihood),
            Pointer.to(new int[]{Nfits})
        );
        
        // Call the kernel function.
        int gridSizeX = (int)Math.ceil((float)Nfits / BlockSize);
        checkResult(cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                BlockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            ));
        checkResult(cuCtxSynchronize());

        // Allocate host output memory and copy the device output to the host.
        Map<String,float[]> result = new HashMap<>();
        float hostParameters[] = new float[PARAMETER_LENGTH * Nfits];
        checkResult(cudaMemcpy(Pointer.to(hostParameters), d_Parameters, PARAMETER_LENGTH * Nfits * Sizeof.FLOAT, cudaMemcpyDeviceToHost));
        float hostCRLBs[] = new float[PARAMETER_LENGTH * Nfits];
        checkResult(cudaMemcpy(Pointer.to(hostCRLBs), d_CRLBs, PARAMETER_LENGTH * Nfits * Sizeof.FLOAT, cudaMemcpyDeviceToHost));
        float hostLogLikelihood[] = new float[PARAMETER_LENGTH * Nfits];
        checkResult(cudaMemcpy(Pointer.to(hostLogLikelihood), d_LogLikelihood, Nfits * Sizeof.FLOAT, cudaMemcpyDeviceToHost));
        
        result.put("Parameters", hostParameters);
        //result.put("CRLB", hostCRLBs);
        //result.put("LogLikelihood", hostLogLikelihood);
        
        cudaFree(d_Parameters);
        cudaFree(d_CRLBs);
        cudaFree(d_LogLikelihood);
        cuCtxDestroy(context);
        System.out.println("count:"+ count++ +" Kernels:" + nKernels + " BlockSize:"+ BlockSize + " GridSize:" + gridSizeX +" Elapsed time in ms: "+(System.currentTimeMillis()-start));
		return result;
    }
	
	private static void checkResult(int cuResult){
        if (cuResult != CUresult.CUDA_SUCCESS)
            throw new CudaException(CUresult.stringFor(cuResult)); 
    }
}
