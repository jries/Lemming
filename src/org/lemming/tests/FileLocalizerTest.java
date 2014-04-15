package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.inputs.FileLocalizer;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

/**
 * Test class for reading localizations from a file. 
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class FileLocalizerTest {
	
	Properties p;
	FileLocalizer fl;

	@Before
	public void setUp() throws Exception {		
		p = new Properties();
		p.load(new FileReader("test.properties"));
	}

	@Test
	public void testCSV() {
		fl = new FileLocalizer(p.getProperty("samples.dir")+"FileLocalizer.csv");
                int frame_count = 0;
                while (fl.hasMoreOutputs()) {
                    fl.newOutput();
                    ++frame_count;
                }
		assertEquals(10, frame_count);
	}

	@Test
	public void testTXT() {
		fl = new FileLocalizer(p.getProperty("samples.dir")+"FileLocalizer.txt");
                int frame_count = 0;
                while (fl.hasMoreOutputs()) {
                    fl.newOutput();
                    ++frame_count;
                }
		assertEquals(10, frame_count);
	}

}
