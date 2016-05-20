package org.lemming.tests;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

import ij.gui.*;
import ij.plugin.OverlayLabels;
import ij.process.FloatPolygon;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.LinkedStore;
import org.lemming.pipeline.Localization;
import org.lemming.plugins.*;
import org.lemming.tools.LemmingUtils;
import org.lemming.interfaces.Element;

import ij.ImagePlus;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

class FitterTest<T extends RealType<T> & NativeType<T>> {

	private static Store store;
	private ImagePlus img;

	FrameElements<T> setUp() {
		// load input image
		//final String filename = System.getProperty("user.home")+"/ownCloud/test.tif";
		final String filename = "H:\\ownCloud\\test.tif";
		img = new ImagePlus(filename);
		Object ip = img.getStack().getPixels(1);
		
		int width = img.getWidth();
		int height = img.getHeight();
		Img<T> slice = LemmingUtils.wrap(ip, new long[]{width, height});
		Frame<T> f = new ImgLib2Frame<>(1, width, height, 150, slice);
		List<Element> locList = new ArrayList<>();
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
		locList.add(new Localization(2L, 20550, 25650, 1859, 1L));
		locList.add(new Localization(3L, 19150, 7050, 1510, 1L));
		locList.add(new Localization(4L, 34400, 14900, 644, 1L));
		locList.add(new Localization(5L, 28650, 3900, 810, 1L));
		locList.add(new Localization(6L, 27750, 7200, 810, 1L));
		locList.add(new Localization(7L, 26100, 11250, 535, 1L));
		locList.add(new Localization(8L, 18600, 17250, 860, 1L));
		locList.add(new Localization(9L, 9900, 28050, 810, 1L));
		locList.add(new Localization(10L, 11100, 7950, 810, 1L));
		locList.add(new Localization(11L, 17250, 9100, 410, 1L));
		locList.add(new Localization(12L, 11850, 7950, 810, 1L));
		locList.add(new Localization(13L, 10000, 24550, 1810, 1L));
		locList.add(new Localization(14L, 9000, 26500, 1300, 1L));
		locList.add(new Localization(15L, 9950, 24650, 1648, 1L));
		locList.add(new Localization(16L, 14700, 27900, 500, 1L));
		locList.add(new Localization(17L, 18500, 32800, 500, 1L));
		locList.add(new Localization(18L, 8855, 10355, 1158, 1L));
		locList.add(new Localization(19L, 10850, 18600, 500, 1L));
		
		FrameElements<T> el = new FrameElements<>(locList, f);
		el.setLast(true);
		
		store = new LinkedStore(19);

		MyCanvas canvas = new MyCanvas(img);
		StackWindow window = new StackWindow(img,canvas);
		window.setVisible(true);
		Overlay overlay = OverlayLabels.createOverlay();
		overlay.drawLabels(true);
		if (overlay.getLabelFont()==null && overlay.getLabelColor()==null) {
			overlay.setLabelColor(Color.white);
			overlay.drawBackgrounds(true);
		}
		overlay.drawNames(false);
		overlay.drawLabels(true);
		img.setOverlay(overlay);
		return el;
	}

	public void showResults(Store store, Color color){
		List<Element> list = new ArrayList<>(store.size());
		while(!store.isEmpty()) {
			Element el = store.poll();
			list.add(el);
			System.out.println(el.toString());
		}

		final Overlay layer = img.getOverlay();
		final FloatPolygon points = LemmingUtils.convertToPoints(list, new Rectangle(0,0,img.getWidth(),img.getHeight()), img.getCalibration().pixelDepth);
		final Roi roi = new PointRoi(points);
		roi.setStrokeColor(color);
		layer.add(roi);
		img.setOverlay(layer);
	}


	public void groundTruth(){
		final Overlay layer = img.getOverlay();
		final double pixelSize=img.getCalibration().pixelDepth;
		final FloatPolygon points = new FloatPolygon();
		points.addPoint(35146.07, 7967.07);
		points.addPoint(20664.84, 25686.89);
		points.addPoint(19174.15, 7070.68);
		points.addPoint(35447.20, 14919.11);
		points.addPoint(28790.34, 3925.55);
		points.addPoint(27807.74, 7305.48);
		points.addPoint(26126.79, 11352.91);
		points.addPoint(18718.43, 17314.28);
		points.addPoint(9963.56, 28163.49);
		points.addPoint(11145.80, 8031.15);
		points.addPoint(17347.55, 9120.66);
		points.addPoint(11869.19, 7995.76);
		points.addPoint(9982.00, 24554.43);
		points.addPoint(9006.98, 26516.96);
		points.addPoint(14804.10, 27889.68);
		points.addPoint(18427.59, 32786.89);
		points.addPoint(8873.59, 10350.40);
		points.addPoint(10875.00, 18569.29);
		for(int i=0;i<points.npoints;i++){
			points.xpoints[i]/=pixelSize;
			points.ypoints[i]/=pixelSize;
		}
		final Roi roi = new PointRoi(points);
		roi.setStrokeColor(Color.yellow);
		layer.add(roi);
		img.setOverlay(layer);
	}


	public static <T extends RealType<T> & NativeType<T>> void main(String[] args) {
		final FitterTest<T> ft = new FitterTest<>();
		final FrameElements<T> el = ft.setUp();

		// Gaussian
		//AbstractModule mf = new GaussianFitter<T>(5,LemmingUtils.readCSV(System.getProperty("user.home")+"/ownCloud/set1-calt.csv"));
		AbstractModule mf = new GaussianFitter<T>(5,LemmingUtils.readCSV("H:\\ownCloud\\set1-calt.csv"));
		mf.setOutput(store);
		mf.processData(el);
		System.out.println("Gaussian\n");
		ft.showResults(store,Color.cyan);

		// Quadratic
		AbstractModule qf = new QuadraticFitter<T>(5);
		qf.setOutput(store);
		qf.processData(el);
		System.out.println("Quadratic\n");
		ft.showResults(store,Color.blue);

		// Symmetric Gaussian 
		AbstractModule sf = new SymmetricGaussianFitter<T>(5);
		sf.setOutput(store);
		sf.processData(el);
		System.out.println("SymmetricGaussian\n");
		ft.showResults(store,Color.green);

		// MLE Java
		System.out.println("MLE\n");
		AbstractModule gf = new MLE_Fitter<>(5, 1152*8);
		gf.setOutput(store);
		gf.processData(el);
		ft.showResults(store,Color.red);
		ft.groundTruth();

		// M2LE Java
//		System.out.println("M2LE\n");
//		AbstractModule m2f = new M2LE_Fitter<>(6, 1152*8,0.9f,728f);
//		gf.setOutput(store);
//		gf.processData(el);
//		ft.showResults(store,Color.orange);

	}

	class MyCanvas extends ImageCanvas {

		private static final long serialVersionUID = 1L;

		public MyCanvas(ImagePlus imp) {
			super(imp);
		}

		@Override
		public void mousePressed(MouseEvent e) {
			int ox = offScreenX(e.getX());
			int oy = offScreenY(e.getY());
			setupScroll(ox, oy);
		}

		@Override
		public void mouseDragged(MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			xMouse = offScreenX(x);
			yMouse = offScreenY(y);
			flags = e.getModifiers();
			scroll(x, y);
		}
	}
}
