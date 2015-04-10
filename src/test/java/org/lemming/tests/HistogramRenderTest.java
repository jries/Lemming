package org.lemming.tests;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.FileLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.outputs.HistogramRender;
import org.lemming.utils.LemMING;

/**
 * Test class for reading localizations from a file and rendering 
 * each localization in a 2D histogram. 
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class HistogramRenderTest {

	FileLocalizer f;
	HistogramRender histo;
	Store<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {		
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		f = new FileLocalizer(p.getProperty("samples.dir")+"HistoRender.csv");
		
		localizations = new QueueStore<Localization>();
		
		f.setOutput(localizations);
	}

	@Test
	public void testShowAll() {		

		// set the number of bins in the x and y dimensions for the histogram to be 1024
		// set the x and y range to be from 0 to 256 to display all localizations
		histo = new HistogramRender(1024, 1024, 0, 256, 0, 256);
		histo.setInput(localizations);
		histo.setTitle("Histogram All");

		// read localizations from the file and render the histogram
		new Thread(f).start();
		new Thread(histo).start();		
		
		while (!localizations.isEmpty()) LemMING.pause(100);
		
		LemMING.pause(2000);
	}

	@Test
	public void testShowRegion() {		
		
		// set the number of bins in the x and y dimensions for the histogram to be 512
		// only display the localizations in the x range from 170 to 195 pixels and in the y range from 115 to 135 pixels 
		histo = new HistogramRender(512, 512, 170, 195, 115, 135);
		histo.setInput(localizations);
		histo.setTitle("Histogram Region");

		// read localizations from the file and render the histogram
		new Thread(f).start();
		new Thread(histo).start();		
		
		while (!localizations.isEmpty()) LemMING.pause(100);
		
		LemMING.pause(2000);
	}

}
