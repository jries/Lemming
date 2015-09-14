package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.*;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;

public class DummyManagerTest {

	private Pipeline pipe;
	private FastStore images;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("");

		images = new FastStore();
		
		DummyImageLoader tif = new DummyImageLoader(10, 5, 128, 128);
		//ImageLoader tif = new ImageLoader(new ImagePlus("/home/ronny/ownCloud/storm/p500ast.tif"));
		tif.setOutput(images);
		
		pipe.add(tif);
		
	
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,images.isEmpty());
	}

}
