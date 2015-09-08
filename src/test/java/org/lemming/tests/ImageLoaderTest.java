package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ImageLoader;
import org.lemming.pipeline.FastStore;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class ImageLoaderTest {

	private ImageLoader tif;
	private FastStore frames;

	
	@Before
	public void setUp() throws Exception {		
		tif = new ImageLoader(new ImagePlus("/Users/ronny/Documents/TubulinAF647.tif"));
		frames = new FastStore();
		tif.setOutput( frames);
	}

	@Test
	public void test() throws InterruptedException {
		long start = System.currentTimeMillis();
		tif.run();
		long end = System.currentTimeMillis();
		System.out.println("Time eleapsed: "+ (end-start));
		assertEquals(9990, frames.getLength());	
		tif.show();
		Thread.sleep(5000);
	}

}
