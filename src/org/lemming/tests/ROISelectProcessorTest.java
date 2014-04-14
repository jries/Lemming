package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import ij.gui.EllipseRoi;
import ij.gui.Roi;

import org.lemming.data.Localization;
import org.lemming.data.NonblockingQueueStore;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.data.XYLocalization;
import org.lemming.inputs.FileLocalizer;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.ROISelectProcessor;
import org.lemming.utils.LemMING;

/**
 * Test class for filtering localizations within a specified region of interest.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class ROISelectProcessorTest {
	
	ROISelectProcessor roi;
	FileLocalizer fl;
	PrintToScreen pts;
	
	@Before
	public void setUp() throws Exception {
		fl = new FileLocalizer("FileLocalizer.csv");
		roi = new ROISelectProcessor(25, 125, 67, 100);
		pts = new PrintToScreen();
	}
	
	@Test
	public void testEmpty() {
                while (fl.hasMoreOutputs()) {
                        roi.process(fl.newOutput());
                }
	}

	@Test
	public void testCircleRoi() {
		Array<Localization> localizations = new Array<Localization>();
		
		// Add some localizations to the input queue
		localizations.put(new XYLocalization(0,0));
		localizations.put(new XYLocalization(100,0));
		localizations.put(new XYLocalization(50,50));
		localizations.put(new XYLocalization(100,100));
		
		Roi circle = new EllipseRoi(15, 15, 85, 85, 1);
		
		roi = new ROISelectProcessor(circle);
                Array<Localizations> result = roi.process(localizations);
		
		assertEquals(result.getLength(), 1);
		assertEquals(result[0].getX(), 50, .001);
	}

}
