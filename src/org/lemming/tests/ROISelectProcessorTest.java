package org.lemming.tests;

import static org.junit.Assert.*;
//import ij.ImageJ;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.input.FileLocalizer;
//import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processor.ROISelectProcessor;

public class ROISelectProcessorTest {
	
	QueueStore<Localization> localizations;
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
		
		assertEquals(localizations.getLength(), 0);
		assertEquals(filteredLocalizations.getLength(), 0);		
	}

}
