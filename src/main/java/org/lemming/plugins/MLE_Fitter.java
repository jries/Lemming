package org.lemming.plugins;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

import javolution.util.FastTable;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.FitterPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Kernel;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.scijava.plugin.Plugin;

public class MLE_Fitter<T extends RealType<T>> extends Fitter<T> {
	
	public static final String NAME = "Maximum Likelihood";

	public static final String KEY = "MLEFITTER";

	public static final String INFO_TEXT = "<html>" + "Maximum likelihood estimation using the NVIDIA CUDA capabilities " + "</html>";

	private static final int PARAMETER_LENGTH = 6;

	private int maxKernels;

	private FastTable<Kernel> kernelList;
	
	private CUdevice device;

	private int kernelSize;

	
	public MLE_Fitter(int windowSize) {
		super(windowSize);
		kernelSize = 2 * size + 1;
		//maxKernels = (int) (40000/Math.pow(kernelSize, 3)*1500);
		maxKernels = 1152*9;
		kernelList = new FastTable<>();
		JCudaDriver.setExceptionsEnabled(true);
 		cuInit(0);
 		device = new CUdevice();
 		cuDeviceGet(device, 0); 
	}

	public void process(FrameElements<T> fe) {
		final List<Element> sliceLocs = fe.getList();
		final double pixelDepth = fe.getFrame().getPixelDepth();
		final RandomAccessible<T> source = Views.extendMirrorSingle(fe.getFrame().getPixels());
		
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			double x = loc.getX().doubleValue() / pixelDepth;
			double y = loc.getY().doubleValue() / pixelDepth;
			
			long xstart = Math.max(0, Math.round(x - size));
			long ystart = Math.max(0, Math.round(y - size));
			long xend = xstart + kernelSize - 1;
			long yend = ystart + kernelSize - 1;
			
			final Interval roi = new FinalInterval(new long[] { xstart, ystart }, new long[] { xend, yend });
			IntervalView<T> interval = Views.interval(source, roi); 
			
			Cursor<T> c = interval.cursor();
			float[] IVal = new float[(int) (kernelSize * kernelSize)];
			int index=0;
			while (c.hasNext()){
				IVal[index++]=c.next().getRealFloat();
			}
			kernelList.add(new Kernel(loc.getID(), loc.getFrame(), roi, IVal));
			if (kernelList.size()>=maxKernels){
				processGPU(pixelDepth);
				kernelList.clear();
			}
		}
		if (fe.isLast()){
			processGPU(pixelDepth);
			kernelList.clear();
			cancel();
			return;
		}
	}
	
	private void processGPU(double pixelDepth){
		GPUBlockThread t = new GPUBlockThread(device, kernelList, kernelSize, kernelList.size(), PARAMETER_LENGTH, "kernel_MLEFit_sigmaxy");
		FutureTask<Map<String, float[]>> ft = new FutureTask<>(t);
		//MLE t = new MLE(kernelList, kernelSize, kernelList.size());
		new Thread(ft).start();
		//Future<Map<String, float[]>> f = singleService.submit(t);
		try {
			Map<String, float[]> res = ft.get();
			float[] par = res.get("Parameters");
			float[] fits = res.get("CRLBs");
			int ksize = kernelList.size();
			for (int i=0;i<ksize;i++){
				long xstart = kernelList.get(i).getRoi().min(0);
				long ystart = kernelList.get(i).getRoi().min(1);
				float x = par[i] + xstart;
				float y = par[ksize+i] + ystart;
				float intensity = par[2*ksize+i];
				float fitI = fits[2*ksize+i];
				float bg = par[3*ksize+i];
				float sx = par[4*ksize+i];
				//float sy = par[5*ksize+i];
				long frame = kernelList.get(i).getFrame();
				long id = kernelList.get(i).getID();
				newOutput(new LocalizationPrecision3D(id, x*pixelDepth, y*pixelDepth, fitI, sx*pixelDepth, 0/*sy*pixelDepth*/, bg, intensity, frame));
			}
		} catch (InterruptedException | ExecutionException | ArrayIndexOutOfBoundsException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public boolean check() {
		return inputs.size() == 1;
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		FrameElements<T> fe = (FrameElements<T>) data;
		process(fe);
		return null;
	}
	
	protected void afterRun() {
		Localization lastLoc = new LocalizationPrecision3D(-1, -1, -1, 0, 0, 0, 0, 1l);
		lastLoc.setLast(true);
		newOutput(lastLoc);
		System.out.println("GPU Fitting done in " + (System.currentTimeMillis() - start) + "ms");
	}
	
	@Override
	public void run() {
		Thread.currentThread().setName("MLE");
		if (!inputs.isEmpty() && !outputs.isEmpty()) {
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = nextInput();
				if (data != null) 
					processData(data);
				else pause(10);
			}
			afterRun();
			return;
		}
		if (!inputs.isEmpty()) {  // no outputs
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = nextInput();
				if (data != null) 
					processData(data);
				else pause(10);
			}
			afterRun();
			return;
		}
		if (!outputs.isEmpty()) { // no inputs
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = processData(null);
				newOutput(data);
			}
			afterRun();
			return;
		}
	}

	@Override
	public List<Element> fit(List<Element> sliceLocs, Frame<T> frame, long windowSize) {
		return null;
	}
	
	private class GPUBlockThread implements Callable<Map<String,float[]>> {
		
		private int sz;
		private int sz2;
		private int nKernels;
		private CUdevice device;
		private List<Kernel> kList;
		private int PARAMETER_LENGTH;
		private String functionName;
		private int count = 0;
		
		private static final float PSFSigma = 1.3f;
		private static final int iterations = 200;
		private static final String ptxFileName = "resources/CudaFit.ptx";
		//private static final String ptxFileName = "resources/GPUgaussMLEv2_noShared.ptx";
		private static final float sharedMemPerBlock = 262144;
		//private static final float sharedMemPerBlock = 15000;


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
	        result.put("CRLBs", hostCRLBs);
	        //result.put("LogLikelihood", hostLogLikelihood);
	        
	        cudaFree(d_Parameters);
	        cudaFree(d_CRLBs);
	        cudaFree(d_LogLikelihood);
	        cuCtxDestroy(context);
	        System.out.println("count:"+ count++ +" Kernels:" + nKernels + " BlockSize:"+ BlockSize + " GridSize:" + gridSizeX +" Elapsed time in ms: "+(System.currentTimeMillis()-start));
			return result;
	    }
		
		private void checkResult(int cuResult){
	        if (cuResult != CUresult.CUDA_SUCCESS)
	            throw new CudaException(CUresult.stringFor(cuResult)); 
	    }
	}
	

	@Plugin(type = FitterFactory.class, visible = true)
	public static class Factory implements FitterFactory {

		private Map<String, Object> settings;
		private FitterPanel configPanel = new FitterPanel();

		@Override
		public String getInfoText() {
			return INFO_TEXT;
		}

		@Override
		public String getKey() {
			return KEY;
		}

		@Override
		public String getName() {
			return NAME;
		}

		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			return settings!=null;
		}


		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public <T extends RealType<T>> Fitter<T> getFitter() {
			final int windowSize = (int) settings.get(FitterPanel.KEY_WINDOW_SIZE);
			return new MLE_Fitter<>(windowSize);
		}

		@Override
		public int getHalfKernel() {
			return size;
		}
	}
}
