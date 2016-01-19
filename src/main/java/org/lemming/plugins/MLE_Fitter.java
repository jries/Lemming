package org.lemming.plugins;

import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javolution.util.FastTable;
import jcuda.driver.CUdevice;
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
import org.lemming.tools.GPUBlockThread;
import org.scijava.plugin.Plugin;

public class MLE_Fitter<T extends RealType<T>> extends Fitter<T> {
	
	public static final String NAME = "Maximum Likelihood";

	public static final String KEY = "MLEFITTER";

	public static final String INFO_TEXT = "<html>" + "Maximum likelihood estimation using the NVIDIA CUDA capabilities " + "</html>";

	private static final int PARAMETER_LENGTH = 5;
	
	private int maxKernels;

	private FastTable<Kernel> kernelList;
	
	private CUdevice device;

	private int kernelSize;

	
	public MLE_Fitter(int windowSize) {
		super(windowSize);
		kernelSize = 2 * size + 1;
		maxKernels = (int) (40000/Math.pow(kernelSize, 3)*1500);
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
		
		if ((kernelList.size()+sliceLocs.size())>maxKernels){
			processGPU();
			kernelList.clear();
		}
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			double x = loc.getX().doubleValue() / pixelDepth;
			double y = loc.getY().doubleValue() / pixelDepth;
			
			long xstart = StrictMath.round(x - size);
			long ystart = StrictMath.round(y - size);
			long xend = xstart + kernelSize;
			long yend = ystart + kernelSize;
			
			final Interval roi = new FinalInterval(new long[] { xstart, ystart }, new long[] { xend, yend });
			IntervalView<T> interval = Views.interval(source, roi); 
			
			Cursor<T> c = interval.cursor();
			int index = 0;
			float[] IVal = new float[(int) (roi.dimension(0) * roi.dimension(1))];
			while (c.hasNext()){
				IVal[index++]=c.next().getRealFloat();
			}
			kernelList.add(new Kernel(loc.getID(), loc.getFrame(), roi, IVal));
		}
		if (fe.isLast()){
			processGPU();
			cancel();
			return;
		}
	}
	
	private void processGPU(){
		ExecutorService singleService = Executors.newSingleThreadExecutor();
		GPUBlockThread t = new GPUBlockThread(device, kernelList, kernelSize, kernelList.size(), PARAMETER_LENGTH, "kernel_MLEFit_sigma");
		Future<Map<String, float[]>> f = singleService.submit(t);
		try {
			Map<String, float[]> res = f.get();
			float[] par = res.get("Parameters");
			for (int i=0;i<kernelList.size();i++){
				long xstart = kernelList.get(i).getRoi().min(0);
				long ystart = kernelList.get(i).getRoi().min(1);
				float x = xstart + par[i*PARAMETER_LENGTH+0];
				float y = ystart + par[i*PARAMETER_LENGTH+1];
				float intensity = par[i*PARAMETER_LENGTH+3];
				long frame = kernelList.get(i).getFrame();
				newOutput(new LocalizationPrecision3D(x, y, 0, 0, 0, 0, intensity, frame));
			}
		} catch (InterruptedException | ExecutionException e) {
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
