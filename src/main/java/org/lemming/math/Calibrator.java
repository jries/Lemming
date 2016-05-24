package org.lemming.math;

import java.awt.Rectangle;
import java.util.concurrent.atomic.AtomicInteger;

import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import ij.util.ThreadUtil;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
* This class handle the calibration by calling the various method of the fitter. Each fit runs on a dedicated thread to allow updating of the GUI.
* 
*/
public class Calibrator {	
	/////////////////////////////
	// Results
	private final double[] zgrid;									// z positions of the slices in the stack
	private final double[] Wx;
	private final double[] Wy;
	private final double[] E;
	/////////////////////////////
    // Parameters from ImageStack
	private final int nSlice;

	/////////////////////////////
	// Input from user
    private final int zstep;
    private int rangeStart, rangeEnd;					// Both ends of the restricted z range and length of the restriction
    private final Rectangle roi;
    
	private final ImageStack is;
	private final BSplines b;
	
	public Calibrator(ImagePlus im, int zstep, Roi r){
		this.is = im.getStack();
		this.zstep = zstep;
    	this.nSlice = im.getNSlices(); 
    	im.getWidth(); 
    	im.getHeight();
    	this.roi = r.getBounds();
	
    	// Initialize arrays
    	zgrid = new double[nSlice];						// z position of the frames
    	Wx = new double[nSlice];						// width in x of the PSF
    	Wy = new double[nSlice];						// width in y of the PSF
    	E = new double[nSlice];
    	b = new BSplines();
	}
	
	// ////////////////////////////////////////////////////////////
	// 1D and 2D fits
	public <T extends RealType<T> & NativeType<T>> void fitStack() {

		Thread[] threads = ThreadUtil.createThreadArray(Runtime.getRuntime().availableProcessors()-1);
		final AtomicInteger ai = new AtomicInteger(0);

		for (int ithread = 0; ithread < threads.length; ithread++) {

			threads[ithread] = new Thread("fit_" + ithread) {
				@Override
				public void run() {
					for (int i = ai.getAndIncrement(); i < nSlice; i = ai.getAndIncrement()) {
						final ImageProcessor ip = is.getProcessor(i + 1);
						final Img<T> theImage = LemmingUtils.wrap(ip.getPixels(), new long[]{is.getWidth(), is.getHeight()});
						final IntervalView<T> view = Views.interval(theImage, new long[]{roi.x,roi.y},  new long[]{roi.x+roi.width, roi.y+roi.height});
						final Gaussian2DFitter<T> gf = new Gaussian2DFitter<>(view, 200, 200);
						final double[] results = gf.fit();
						final Gradient<T> cf = new Gradient<>(view, 0, 7);
						final double[] ce = cf.fit();
						if (ce!=null)
							E[i]=ce[2];
						if (results!=null){
							Wx[i]=results[2];
							Wy[i]=results[3];
						}
					}
				}
			};
		}
		ThreadUtil.startAndJoin(threads);

		createZgrid(zgrid, 0);
		fixCurve(Wx);
		fixCurve(Wy);
		fixCurve(E);
		b.plotPoints(zgrid, Wx, Wy, "Width of Elliptical Gaussian");
	}	
	
	private static void fixCurve(double[] d) {
		for (int i=1 ; i<d.length-1;i++)
			if (d[i]<0.1) d[i]=(d[i-1]+d[i+1])/2;
	}
	
	public void fitBSplines(final int rStart, final int rEnd) {
		calculateRange(rStart, rEnd);
		int arraySize = rangeEnd - rangeStart + 1;
		final double[] rangedZ = new double[arraySize];
		final double[] rangedWx = new double[arraySize];
		final double[] rangedWy = new double[arraySize];
		final double[] rangedE = new double[arraySize];

		System.arraycopy(zgrid, rangeStart, rangedZ, 0, arraySize);
		System.arraycopy(Wx, rangeStart, rangedWx, 0, arraySize);
		System.arraycopy(Wy, rangeStart, rangedWy, 0, arraySize);
		System.arraycopy(E, rangeStart, rangedE, 0, arraySize);

		Thread t = new Thread(new Runnable() {

			@Override
			public void run() {
	            b.init(rangedZ, rangedWx, rangedWy, rangedE);
	
	            // Display result
	            b.plotWxWyFitCurves();
	            //b.plot(rangedE, "ellipticity");
			}
        });
		t.start();
		try {
			t.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	//////////////////////////////////////////////////////////////
	// Save
	public void saveCalib(String path){
		b.saveAsCSV(path);
	}
	
	//////////////////////////////////////////////////////////////
	// Misc functions
	
	private void createZgrid(double[] z, int offset){
		for(int i=0;i<z.length;i++)
			z[i] = (i-offset)*zstep;
	}
	
	private void calculateRange(final int rStart, final int rEnd){
		double minStart = Math.abs(rStart-zgrid[0]);
		double minEnd = Math.abs(rEnd-zgrid[nSlice-1]);
		int iStart = 0;
		int iEnd = nSlice-1;
		for(int i=1;i<nSlice;i++){
			if(Math.abs(rStart-zgrid[i])<minStart){
				minStart = Math.abs(rStart-zgrid[i]);
				iStart = i;
			}
			if(Math.abs(rEnd-zgrid[nSlice-1-i])<minEnd){
				minEnd = Math.abs(rEnd-zgrid[nSlice-1-i]);
				iEnd = nSlice-1-i;
			}
		}
		this.rangeStart = iStart;
		this.rangeEnd = iEnd;
	}

	public double[] getZgrid() {
		return zgrid;
	}

	public void closePlotWindows() {
		b.closePlotWindows();
	}
}













