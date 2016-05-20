package org.lemming.tests;

import java.awt.Rectangle;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

import org.lemming.interfaces.Frame;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.plugins.NMSFastMedian;
import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.PointRoi;
import ij.gui.StackWindow;
import ij.plugin.FileInfoVirtualStack;
import ij.process.FloatPolygon;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

class NMSFastMedianTest<T extends NativeType<T> & RealType<T>> {
		
	private void setUp() {
		final File file = new File(System.getProperty("user.home")+"/ownCloud/exp-images.tif");
		final ImagePlus loc_im  = FileInfoVirtualStack.openVirtual(file.getAbsolutePath());
		final double pixelSize = loc_im.getCalibration().pixelDepth;
		final StackWindow previewerWindow = new StackWindow(loc_im, loc_im.getCanvas());
		previewerWindow.setImage(loc_im);
		previewerWindow.getCanvas().fitToWindow();
		previewerWindow.repaint();
		previewerWindow.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				if (previewerWindow != null) previewerWindow.close();
			}
		});
		final ImagePlus img = previewerWindow.getImagePlus();
		img.setSlice(150);
		img.setDisplayRange(0, 8000);
		NMSFastMedian<T> preprocessor =  new NMSFastMedian<T>(100, true, 5, 5);
		List<Double> cameraProperties = LemmingUtils.readCameraSettings(System.getProperty("user.home")+"/camera.props");
		final Double offset = cameraProperties.get(0);
		final Double em_gain = cameraProperties.get(1);
		final Double conversion = cameraProperties.get(2);
		final int frameNumber = img.getSlice();
		final ImageStack stack = img.getStack();
		final int stackSize = stack.getSize();
		final Queue<Frame<T>> list = new ArrayDeque<Frame<T>>();
		final int start = frameNumber/preprocessor.getNumberOfFrames()*preprocessor.getNumberOfFrames();
		double adu, im2phot;
		Frame<T> origFrame=null;
		
		for (int i = start; i < start + preprocessor.getNumberOfFrames(); i++) {
			if (i < stackSize) {
				Object ip = stack.getPixels(i+1);
				Img<T> curImage = LemmingUtils.wrap(ip, new long[]{stack.getWidth(), stack.getHeight()});
				final Cursor<T> it = curImage.cursor();
				while(it.hasNext()){
					it.fwd();
					adu = Math.max((it.get().getRealDouble()-offset), 0);
					im2phot = adu*conversion/em_gain;
					it.get().setReal(im2phot);
				}
				Frame<T> curFrame = new ImgLib2Frame<T>(i, (int) curImage.dimension(0), (int) curImage.dimension(1), pixelSize, curImage);
				if (i==frameNumber) origFrame=curFrame;
				list.add(curFrame);
			}
		}
		if (origFrame==null) origFrame=list.peek();
		
		final Frame<T> result = preprocessor.preProcess(list,true);
		final FrameElements<T> detResults = preprocessor.detect(LemmingUtils.substract(result,origFrame));
		if (detResults.getList().isEmpty()) return;
		final FloatPolygon points = LemmingUtils.convertToPoints(detResults.getList(), new Rectangle(0,0,img.getWidth(),img.getHeight()), pixelSize);
		final PointRoi roi = new PointRoi(points);
		previewerWindow.getImagePlus().setRoi(roi);
	}

	@SuppressWarnings("rawtypes")
	public static void main(String[] args) {
		NMSFastMedianTest mt = new NMSFastMedianTest();
		mt.setUp();
	}

}
