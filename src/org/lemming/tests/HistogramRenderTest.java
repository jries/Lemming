package org.lemming.tests;

import ij.ImageJ;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.input.FileLocalizer;
import org.lemming.outputs.HistogramRender;

public class HistogramRenderTest {

	FileLocalizer f;
	HistogramRender histo;
	Store<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {		
		ImageJ.main(new String[]{});
		
		f = new FileLocalizer("HistoRender.csv");
		histo = new HistogramRender();
		
		localizations = new QueueStore<Localization>();
		
		f.setOutput(localizations);
		histo.setInput(localizations);		
	}

	@Test
	public void test() {
		
		// set the x,y bin widths for the histogram
		int xBins = 1024;
		int yBins = 1024;
		histo.setBins(xBins, yBins);

		// set the x,y range of localization values to show in the histogram
		//double xmin = 0, ymin = 0, xmax = 256, ymax = 256; // show the all localizations
		double xmin = 0, ymin = 0, xmax = 256, ymax = 256; // show the all localizations
		//double xmin = 170, ymin = 115, xmax = 195, ymax = 135; // zoom in for a specific region
		histo.setRange(xmin, xmax, ymin, ymax);

		// read localizations from the file and render the histogram
		new Thread(f).start();
		new Thread(histo).start();		
		
		while (true){}
	}

}
