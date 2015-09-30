package org.lemming.modules;

import java.awt.Rectangle;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.MultiRunModule;

import ij.gui.Roi;

public abstract class Fitter<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {

	private long start;
	protected int size;
	private int queueSize;
	private AtomicInteger counter = new AtomicInteger(0);

	public Fitter(int queueSize, int windowSize) {
		this.size = windowSize;
		this.queueSize = queueSize;
	}

	public int getWindowSize(){
		return size;
	}

	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}
	

	@SuppressWarnings("unchecked")
	@Override
	public Element process(Element data) {
		FrameElements<T> fe = (FrameElements<T>) data;
		if (fe == null)
			return null;

		if (fe.isLast()) {
			if (!inputs.get(iterator).isEmpty()){
				inputs.get(iterator).put(fe);
				return null;
			}
			cancel();
			process1(fe);
		}
		
		process1(fe);
		return null;
	}

	private void process1(FrameElements<T> data) {
		List<Element> res = fit( data.getList(), data.getFrame().getPixels(), size, data.getFrame().getFrameNumber());
		counter.addAndGet(res.size());
		for (Element el : res)
			newOutput(el);
	}
	
	public abstract List<Element> fit(List<Element> sliceLocs, RandomAccessibleInterval<T> pixels, long windowSize, long frameNumber);

	@SuppressWarnings({ })
	@Override
	protected void afterRun() {

		FittedLocalization lastLoc = new FittedLocalization(0, -1, -1, 0, -1, -1) ;
		lastLoc.setLast(true);
		newOutput(lastLoc);
		System.out.println("Fitting of "+ counter +" elements done in " + (System.currentTimeMillis() - start)+"ms");
	}

	@Override
	public boolean check() {
		return inputs.size()==1 && outputs.size()>=1;
	}
	
	protected static Roi cropRoi(Rectangle imageRoi, Rectangle curRect) {
		double x1 = curRect.getMinX() < imageRoi.getMinX() ? imageRoi.getMinX() : curRect.getMinX();
		double y1 = curRect.getMinY() < imageRoi.getMinY() ? imageRoi.getMinY() : curRect.getMinY();
		double x2 = curRect.getMaxX() > imageRoi.getMaxX() ? imageRoi.getMaxX() : curRect.getMaxX();
		double y2 = curRect.getMaxY() > imageRoi.getMaxY() ? imageRoi.getMaxY() : curRect.getMaxY();
		return new Roi(x1,y1,x2-x1,y2-y1);
	}


}
