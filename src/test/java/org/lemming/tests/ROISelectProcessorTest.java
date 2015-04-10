package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import ij.gui.EllipseRoi;
import ij.gui.Roi;

import org.lemming.data.NonblockingQueueStore;
import org.lemming.data.QueueStore;
import org.lemming.data.XYLocalization;
import org.lemming.inputs.FileLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.ROISelectProcessor;
import org.lemming.utils.LemMING;

/**
 * Test class for filtering localizations within a specified region of interest.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
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
	public void testEmpty() {
		new Thread(fl).start();
		new Thread(roi).start();
		new Thread(pts).start();

		LemMING.pause(100);
		
		assertTrue(localizations.isEmpty());
		assertTrue(filteredLocalizations.isEmpty());
	}

	@Test
	public void testCircleRoi() {
		localizations = new NonblockingQueueStore<Localization>();
		
		// Add some localizations to the input queue
		localizations.put(new XYLocalization(0,0));
		localizations.put(new XYLocalization(100,0));
		localizations.put(new XYLocalization(50,50));
		localizations.put(new XYLocalization(100,100));
		
		Roi circle = new EllipseRoi(15, 15, 85, 85, 1);
		
		roi = new ROISelectProcessor(circle);
		roi.setInput(localizations);
		roi.setOutput(filteredLocalizations);
		
		roi.run();
		
		assertTrue(localizations.isEmpty());
		assertEquals(filteredLocalizations.getLength(), 1);
		assertEquals(filteredLocalizations.get().getX(), 50, .001);
	}

}
