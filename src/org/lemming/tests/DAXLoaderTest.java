package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ShortProcessor;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.DAXLoader;

public class DAXLoaderTest {

	Frame f;
	DAXLoader dax;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		
	
		dax = new DAXLoader("daxSample.dax");
		frames = new QueueStore<Frame>();
		
		dax.setOutput(frames);
	}

	@Test
	public void test() {
		dax.run();
		
		assertEquals(860, frames.getLength());
		
		ImageStack stack = new ImageStack(dax.width, dax.height);
		while (!frames.isEmpty()) {
			stack.addSlice(new ShortProcessor(dax.width, dax.height, (short[])frames.get().getPixels(), null));
		}		
		new ImagePlus(dax.daxFilename, stack).show();

		
		//int cnt = 0;
		//short[] pixels;
		//while (!frames.isEmpty()){
		//	f = frames.get();
		//	pixels = (short[]) f.getPixels();
		//	assertEquals(256*256, pixels.length);
		//	System.out.println("Frame: "+ ++cnt + ", Pixel(0,0): " + pixels[0]);
		//}
		
		//dax.show();
		while(true){}
		
	}
}
