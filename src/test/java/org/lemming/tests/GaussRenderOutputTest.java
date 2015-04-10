package org.lemming.tests;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.FileLocalizer;
import org.lemming.interfaces.Localization;
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
	
	Localizer fl;
	GaussRenderOutput gro;
	QueueStore<Localization> localizations; 
	
	@Before
	public void setUp() throws Exception {		
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		fl = new FileLocalizer(p.getProperty("samples.dir")+"FileLocalizer.txt");
		localizations = new QueueStore<Localization>();
		gro = new GaussRenderOutput();
		
		fl.setOutput(localizations);
		gro.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(fl).start();
		new Thread(gro).start();
		
		LemMING.pause(2000);
	}

}
