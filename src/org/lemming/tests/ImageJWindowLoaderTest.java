package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import ij.IJ;
import ij.ImageJ;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.inputs.ImageJWindowLoader;
import org.lemming.inputs.ScifioLoader;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a TIF file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class ImageJWindowLoaderTest {

	Frame f;
	ImageJWindowLoader tif;
	
	@Before
	public void setUp() throws Exception {
		
		new ImageJ();
		
		IJ.run( "MRI Stack (528K)" );
		
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ImageJWindowLoader();
	}

	@Test
	public void test() {
                int frame_count = 0;
                while (tif.hasMoreOutputs()) {
                    tif.newOutput();
                    ++frame_count;
                }
		assertEquals(27, frame_count);		
	}

}
