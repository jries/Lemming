package org.lemming.tests;

import java.util.ArrayList;
import java.util.List;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.LocalizationInterface;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.LinkedStore;
import org.lemming.pipeline.Localization;
import org.lemming.plugins.GaussianFitter;
import org.lemming.plugins.GradientFitter;
import org.lemming.plugins.MLE_Fitter;
import org.lemming.plugins.QuadraticFitter;
import org.lemming.plugins.SymmetricGaussianFitter;
import org.lemming.tools.LemmingUtils;
import org.lemming.interfaces.Element;

import ij.ImagePlus;
import net.imglib2.FinalInterval;
import net.imglib2.FinalRealInterval;
import net.imglib2.Interval;
import net.imglib2.img.Img;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class FitterTest<T extends RealType<T> & NativeType<T>> {

	private static Store store;
	private static int halfKernel=6;
	private static double pixelDepth=138.0;

	public FrameElements<T> setUp() {
		// load input image
		final String filename = "D:/ownCloud/test.tif";
		ImagePlus img = new ImagePlus(filename);
		Object ip = img.getStack().getPixels(1);
		
		int width = img.getWidth();
		int height = img.getHeight();
		Img<T> slice = LemmingUtils.wrap(ip, new long[]{width, height});
		Frame<T> f = new ImgLib2Frame<T>(1, width, height, 150, slice);
		List<Element> locList = new ArrayList<Element>();
		/*1, 1, 35146.07, 7967.07, 45.07, 810.00
		2, 1, 20664.84, 25686.89, 124.50, 1859.52
		3, 1, 19174.15, 7070.68, 116.55, 1534.96
		4, 1, 35447.20, 14919.11, 243.77, 644.63
		5, 1, 28790.34, 3925.55, 166.71, 1043.79
		6, 1, 27807.74, 7305.48, 191.37, 1010.58
		7, 1, 26126.79, 11352.91, 230.95, 2647.54
		8, 1, 18718.43, 17314.28, 123.68, 2692.50
		9, 1, 9963.56, 28163.49, 83.00, 1311.65
		10, 1, 11145.80, 8031.15, 250.37, 1135.84
		11, 1, 17347.55, 9120.66, 190.93, 1471.17
		12, 1, 11869.19, 7995.76, 224.31, 1221.05
		13, 1, 9982.00, 24554.43, 118.51, 1879.31
		14, 1, 9006.98, 26516.96, 109.45, 1305.34
		15, 1, 9945.02, 24671.93, 118.01, 1642.76
		16, 1, 14804.10, 27889.68, 163.68, 1863.61
		17, 1, 18427.59, 32786.89, 118.43, 1568.37
		18, 1, 8873.59, 10350.40, 109.07, 1158.49
		19, 1, 10875.00, 18569.29, 156.61, 2059.55
		*/
		locList.add(new Localization(1L, 35100, 7950, 810, 1L));
		locList.add(new Localization(2L, 20550, 25650, 859, 1L));
		locList.add(new Localization(3L, 19050, 7050, 810, 1L));
		locList.add(new Localization(4L, 34800, 14400, 810, 1L));
		locList.add(new Localization(5L, 28650, 3900, 810, 1L));
		locList.add(new Localization(6L, 27750, 7200, 810, 1L));
		locList.add(new Localization(7L, 26100, 11250, 535, 1L));
		locList.add(new Localization(8L, 18600, 17250, 860, 1L));
		locList.add(new Localization(9L, 9900, 28050, 810, 1L));
		locList.add(new Localization(10L, 11100, 7950, 810, 1L));
		locList.add(new Localization(11L, 17250, 9100, 410, 1L));
		locList.add(new Localization(12L, 11850, 7950, 810, 1L));
		locList.add(new Localization(13L, 9900, 24602, 810, 1L));
		locList.add(new Localization(14L, 9000, 26500, 645, 1L));
		locList.add(new Localization(15L, 9950, 24600, 648, 1L));
		locList.add(new Localization(16L, 14700, 27900, 500, 1L));
		locList.add(new Localization(17L, 18500, 32800, 500, 1L));
		locList.add(new Localization(18L, 8850, 10350, 810, 1L));
		locList.add(new Localization(19L, 10850, 18600, 500, 1L));
		
		FrameElements<T> el = new FrameElements<T>(locList, f);
		el.setLast(true);
		
		store = new LinkedStore(19);
		return el;
	}

	public static <T extends RealType<T> & NativeType<T>> void main(String[] args) {
		final FitterTest<T> ft = new FitterTest<T>();
		final FrameElements<T> el = ft.setUp();
		NearestNeighborInterpolatorFactory< FloatType > factory = new NearestNeighborInterpolatorFactory< FloatType >();
		for (Element peak:el.getList()){
			final LocalizationInterface loc = (LocalizationInterface) peak;
			
			double x = loc.getX().doubleValue()/pixelDepth;
			double y = loc.getY().doubleValue()/pixelDepth;

			final FinalRealInterval roi = new FinalRealInterval(new double[] { x - halfKernel, y - halfKernel }, new double[] { Math.ceil(x + halfKernel),
					Math.ceil(y + halfKernel) });
		}
		// Gaussian
		AbstractModule mf = new GaussianFitter<T>(halfKernel,LemmingUtils.readCSV("D:/ownCloud/set1-calb.csv"));
		mf.setOutput(store);
		mf.processData(el);
		System.out.println("Gaussian\n");
		while(!store.isEmpty()) 
			System.out.println(store.poll().toString());
		// Quadratic 
		AbstractModule qf = new QuadraticFitter<T>(halfKernel);
		qf.setOutput(store);
		qf.processData(el);
		System.out.println("Quadratic\n");
		while(!store.isEmpty()) 
			System.out.println(store.poll().toString());
		// Symmetric Gaussian 
		AbstractModule sf = new SymmetricGaussianFitter<T>(halfKernel);
		sf.setOutput(store);
		sf.processData(el);
		System.out.println("SymmetricGaussian\n");
		while(!store.isEmpty()) 
			System.out.println(store.poll().toString());
		// Gradient 
		AbstractModule gf = new GradientFitter<T>(halfKernel,LemmingUtils.readCSV("D:/ownCloud/set1-calb.csv"));
		gf.setOutput(store);
		gf.processData(el);
		System.out.println("Gradient\n");
		while(!store.isEmpty()) 
			System.out.println(store.poll().toString());
		//MLE
		AbstractModule mlf = new MLE_Fitter<T>(halfKernel);
		mlf.setOutput(store);
		mlf.preview(el);
		System.out.println("MLE\n");
		while(!store.isEmpty()) 
			System.out.println(store.poll().toString());
		return;
	}

}
