package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ScifioLoader;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class ScifioLoaderTest {

	Frame f;
	ScifioLoader tif;
	
	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ScifioLoader(p.getProperty("samples.dir")+"eye.tif");
	}

	@Test
	public void test() {
                int frame_count = 0;
                while (tif.hasMoreOutputs()) {
                    tif.newOutput();
                    ++frame_count;
                }
		assertEquals(41, frames.getLength());		
	}

}
