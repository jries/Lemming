package org.lemming.modules;

import ij.gui.Roi;
import ij.process.ImageProcessor;

import java.util.List;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.GaussianFitterZ;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.Localization;

public class AstigFitter<T extends RealType<T>, F extends Frame<T>> extends Fitter<T, F> {

	private double[] params;

	public AstigFitter(int queueSize, long windowSize, double[] params ) {
		super(queueSize, windowSize);
		this.params = params;
	}

	@Override
	public void fit(List<Element> sliceLocs, RandomAccessibleInterval<T> pixels, long windowSize) {
		ImageProcessor ip = ImageJFunctions.wrap(pixels,"").getProcessor();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			final Roi origroi = new Roi(loc.getX() - size, loc.getY() - size, 2*size+1, 2*size+1);
			final Roi roi = new Roi(ip.getRoi().intersection(origroi.getBounds()));
			GaussianFitterZ gf = new GaussianFitterZ(ip, roi, 3000, 1000, params);
			double[] result = null;
			result = gf.fit();
			if (result!= null)
				newOutput(new FittedLocalization(loc.getID(),loc.getFrame(), result[0], result[1], result[2], 0, 0));			
		}
	}

}
