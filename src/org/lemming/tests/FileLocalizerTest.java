package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.inputs.FileLocalizer;
import org.lemming.outputs.PrintToScreen;

public class FileLocalizerTest {
	
	FileLocalizer fl;
	PrintToScreen ps;
	
	QueueStore<Localization> localizations;

	@Before
	public void setUp() throws Exception {
		fl = new FileLocalizer("FileLocalizer.csv");
		ps = new PrintToScreen();
		
		localizations = new QueueStore<Localization>();
		
		fl.setOutput(localizations);
		ps.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(fl).start();
		
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		assertEquals(10, localizations.getLength());

		new Thread(ps).start();
		
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		assertEquals(localizations.getLength(), 0);
	}

}
