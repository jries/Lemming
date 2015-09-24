package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ImageGenerator;
import org.lemming.pipeline.FastStore;

@SuppressWarnings("rawtypes")
public class ImageGeneratorTest {

	private ImageGenerator tif;
	private FastStore frames;
	private int numRuns = 1;
	
	@Before
	public void setUp() throws Exception {		
		tif = new ImageGenerator(600,600,numRuns,15,10000,3,15,8);
		frames = new FastStore();
		tif.setOutput( frames);
	}

	@Test
	public void test() throws InterruptedException {
		long start = System.currentTimeMillis();
		tif.run();
		long end = System.currentTimeMillis();
		System.out.println("Time elapsed: "+ (end-start));
		assertEquals(numRuns, frames.getLength());	
		tif.show();
		Thread.sleep(10000);
	}

}
