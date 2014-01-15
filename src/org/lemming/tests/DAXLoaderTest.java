package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ShortProcessor;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.DAXLoader;
import org.lemming.utils.LemMING;

/**
 * Test class to load data from a DAX file format. 
 * NOTE: a test123.DAX files requires a test123.INF file to be located in the same folder.
 * The INF file contains the information for this DAX file, eg. # of frames, frame width & height, etc...
 * 
 * @author Joe Borbely, Thomas Pengo
 *
 */
public class DAXLoaderTest {

	DAXLoader dax;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {		
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
	
		dax = new DAXLoader(p.getProperty("samples.dir")+"daxSample.dax");
		frames = new QueueStore<Frame>();
		
		dax.setOutput(frames);
	}

	@Test
	public void test() {
		dax.run();
		
		assertEquals(57, frames.getLength());
		
		ImageStack stack = new ImageStack(dax.width, dax.height);
		while (!frames.isEmpty()) {
			stack.addSlice(new ShortProcessor(dax.width, dax.height, (short[])frames.get().getPixels(), null));
		}		
		new ImagePlus("DAX test", stack).show();
		
		LemMING.pause(2000);		
	}
}
