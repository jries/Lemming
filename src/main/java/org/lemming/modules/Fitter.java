package org.lemming.modules;

import java.awt.Rectangle;
import java.util.List;
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

	public Fitter(int halfkernel) {
		Fitter.size = halfkernel;
	}
	
	public static int getHalfKernel() {
		return size;
	}

	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public boolean check() {
		return inputs.size() == 1;
	}

	protected static Roi cropRoi(Rectangle imageRoi, Rectangle curRect) {
		double x1 = curRect.getMinX() < imageRoi.getMinX() ? imageRoi.getMinX() : curRect.getMinX();
		double y1 = curRect.getMinY() < imageRoi.getMinY() ? imageRoi.getMinY() : curRect.getMinY();
		double x2 = curRect.getMaxX() > imageRoi.getMaxX() ? imageRoi.getMaxX() : curRect.getMaxX();
		double y2 = curRect.getMaxY() > imageRoi.getMaxY() ? imageRoi.getMaxY() : curRect.getMaxY();
		return new Roi(x1, y1, x2 - x1, y2 - y1);
	}

	public abstract List<Element> fit(List<Element> sliceLocs, Frame<T> frame, long windowSize) ;
}
