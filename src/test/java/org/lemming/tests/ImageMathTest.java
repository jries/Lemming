package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.SaveImages;
import org.lemming.modules.SaveLocalizations;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.FastMedianFilter;
import org.lemming.plugins.NMSDetector;
import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class ImageMathTest {

	private ImageMath im;
	private Manager pipe;
	private ImageLoader tif;

	private FastMedianFilter fmf;
	private Map<Integer, Store> map;
	private NMSDetector det;
	private SaveLocalizations saver;
	private SaveImages si;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		final ImagePlus image = new ImagePlus("D:/Images/test81000.tif");
		tif = new ImageLoader<>(image,LemmingUtils.readCameraSettings("camera.props"));;
		pipe.add(tif);
	
		fmf = new FastMedianFilter(100, true);
		pipe.add(fmf);
		
		im = new ImageMath(100);
		im.setOperator(ImageMath.operators.SUBSTRACTION);
		pipe.add(im);
		
		det = new NMSDetector(700, 7);
		pipe.add(det);
		
		saver = new SaveLocalizations(new File("D:/Images/test.csv"));
		pipe.add(saver);
		
		si = new SaveImages("D:/Images/test.tif");
		pipe.add(si);
		
		pipe.linkModules(tif, fmf, true, image.getStackSize());
		pipe.linkModules(tif, im);
		pipe.linkModules(fmf, im);
		pipe.linkModules(fmf, si);
		pipe.linkModules(im, det);
		pipe.linkModules(det, saver, false, 128);
		map = pipe.getMap();
		
	}

	@Test
	public void test() {
		pipe.run();
		//System.out.println(im.getProcessingTime());	
		assertEquals(true,map.values().iterator().next().isEmpty());
	}

}
