package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Before;
import org.junit.Test;

import ij.gui.EllipseRoi;
import ij.gui.Roi;

import org.lemming.data.Localization;
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
		Roi circle = new EllipseRoi(15, 15, 85, 85, 1);
		roi = new ROISelectProcessor(circle);
                assertNull(roi.process(new XYLocalization(0,0)));
                assertNull(roi.process(new XYLocalization(100,0)));
                assertNull(roi.process(new XYLocalization(100,100)));
                
                Localization kept = new XYLocalization(50,50);
                assertEquals(kept, roi.process(kept));
	}

}
