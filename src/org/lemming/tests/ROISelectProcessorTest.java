package org.lemming.tests;

import static org.junit.Assert.*;
import ij.IJ;
import ij.ImageJ;
import ij.gui.EllipseRoi;
import ij.gui.Roi;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.NonblockingQueueStore;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.data.XYLocalization;
import org.lemming.input.FileLocalizer;
import org.lemming.outputs.GaussRenderOutput;
//import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processor.ROISelectProcessor;

public class ROISelectProcessorTest {
	
	Store<Localization> localizations;
	QueueStore<Localization> filteredLocalizations;
	ROISelectProcessor roi;
	FileLocalizer fl;
	PrintToScreen pts;
	
	@Before
	public void setUp() throws Exception {
		
		fl = new FileLocalizer("FileLocalizer.csv");
		localizations = new QueueStore<Localization>();
		filteredLocalizations = new QueueStore<Localization>();
		roi = new ROISelectProcessor(25, 125, 67, 100);
		pts = new PrintToScreen();
		
		fl.setOutput(localizations);
		roi.setInput(localizations);
		roi.setOutput(filteredLocalizations);
		pts.setInput(filteredLocalizations);
	}
	
	@Test
	public void test() {
		new Thread(fl).start();
		new Thread(roi).start();
		new Thread(pts).start();

		try {
			Thread.sleep(100);			
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		assertTrue(localizations.isEmpty());
		assertTrue(filteredLocalizations.isEmpty());
	}

	@Test
	public void testCircleRoi() {
		// Create non-blocking queue
		localizations = new NonblockingQueueStore<Localization>();
		
		// Add some localizations to the input queue
		localizations.put(new XYLocalization(0,0));
		localizations.put(new XYLocalization(100,0));
		localizations.put(new XYLocalization(50,50));
		localizations.put(new XYLocalization(100,100));
		
		// Set the region
		Roi circle = new EllipseRoi(15, 15, 85, 85, 1);
		
		// Show it
		GaussRenderOutput g = new GaussRenderOutput();
		g.setInput(localizations);
		//g.run();
		
		//IJ.getImage().setRoi(circle);
		
		// Create the filter
		roi = new ROISelectProcessor(circle);
		roi.setInput(localizations);
		roi.setOutput(filteredLocalizations);
		
		// Run it
		roi.run();
		
		// Check the output
		assertTrue(localizations.isEmpty());
		assertEquals(filteredLocalizations.getLength(), 1);
		assertEquals(filteredLocalizations.get().getX(), 50, .001);
	}

}
