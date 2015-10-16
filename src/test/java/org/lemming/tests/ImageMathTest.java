package org.lemming.tests;

import static org.junit.Assert.*;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.ImageMath;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.FastMedianFilter;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class ImageMathTest {

	private ImageMath im;
	private Manager pipe;
	private ImageLoader tif;

	private FastMedianFilter fmf;
	private Map<Integer, Store> map;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		tif = new ImageLoader(new ImagePlus("/Users/ronny/Documents/TubulinAF647.tif"));
		pipe.add(tif);
	
		fmf = new FastMedianFilter(50, true);
		pipe.add(fmf);
		
		im = new ImageMath();
		im.setOperator(ImageMath.operators.SUBSTRACTION);
		pipe.add(im);
		
		pipe.linkModules(tif, fmf, true);
		pipe.linkModules(tif, im);
		pipe.linkModules(fmf, im);
		map = pipe.getMap();
		
	}

	@Test
	public void test() {
		pipe.run();
		//System.out.println(im.getProcessingTime());	
		assertEquals(true,map.values().iterator().next().isEmpty());
	}

}
