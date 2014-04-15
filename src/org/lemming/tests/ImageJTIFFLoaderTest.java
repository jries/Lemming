package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.queue.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.inputs.ScifioLoader;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class ImageJTIFFLoaderTest {

	Frame f;
	ImageJTIFFLoader tif;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ImageJTIFFLoader(p.getProperty("samples.dir")+"eye.tif");
	}

	@Test
	public void test() {
                int frame_count = 0;
                tif.beforeRun();
                while (tif.hasMoreOutputs()) {
                    tif.newOutput();
                    ++frame_count;
                }
                tif.afterRun();
		assertEquals(41, frame_count);		
	}

}
