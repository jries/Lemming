package org.lemming.tests;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.inputs.FileLocalizer;
import org.lemming.interfaces.ImageLocalizer;
import org.lemming.interfaces.Localizer;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.utils.LemMING;

/**
 * Test class for reading localizations from a file and rendering 
 * each localization as a 2D Gaussian point-spread function. 
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class GaussRenderOutputTest {
	
	FileLocalizer fl;
	GaussRenderOutput gro;
	
	@Before
	public void setUp() throws Exception {		
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		fl = new FileLocalizer(p.getProperty("samples.dir")+"FileLocalizer.txt");
		gro = new GaussRenderOutput();
	}

	@Test
	public void test() {
                while (fl.hasMoreOutputs()) {
                    gro.process(fl.newOutput());
                }
	}

}
