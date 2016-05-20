package org.lemming.modules;

import java.awt.Rectangle;
import java.util.List;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.pipeline.AbstractModule;
import ij.gui.Roi;

/**
 * base class for all Fitter plug-ins
 * 
 * @author Ronny Sczech
 *
 * @param <T> - data type
 */
public abstract class Fitter<T extends RealType<T>> extends AbstractModule {

	protected static int size;

	protected Fitter(int halfkernel) {
		size = halfkernel;
	}
	
	public int getHalfKernel() {
		return size;
	}

	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public boolean check() {
		return inputs.size() == 1;
	}

	public static Roi cropRoi(Rectangle imageRoi, Rectangle curRect) {
		double x1 = curRect.getMinX() < imageRoi.getMinX() ? imageRoi.getMinX() : curRect.getMinX();
		double y1 = curRect.getMinY() < imageRoi.getMinY() ? imageRoi.getMinY() : curRect.getMinY();
		double x2 = curRect.getMaxX() > imageRoi.getMaxX() ? imageRoi.getMaxX() : curRect.getMaxX();
		double y2 = curRect.getMaxY() > imageRoi.getMaxY() ? imageRoi.getMaxY() : curRect.getMaxY();
		return new Roi(x1, y1, x2 - x1, y2 - y1);
	}
	
	public static Interval cropInterval(long[] imageMin, long[] imageMax, long[] curMin, long[] curMax ){
		long x1 = curMin[0] < imageMin[0] ? imageMin[0] : curMin[0];
		long y1 = curMin[1] < imageMin[1] ? imageMin[1] : curMin[1];
		long x2 = curMax[0] > imageMax[0] ? imageMax[0] : curMax[0];
		long y2 = curMax[1] > imageMax[1] ? imageMax[1] : curMax[1];	
		return new FinalInterval(new long[] { x1, y1 }, new long[]{ x2, y2 });
	}

	protected abstract List<Element> fit(List<Element> sliceLocs, Frame<T> frame, long windowSize) ;
}
