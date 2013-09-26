package org.lemming.tests;

import ij.ImageJ;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.input.FileLocalizer;
import org.lemming.interfaces.Localizer;
import org.lemming.outputs.GaussRenderOutput;

public class GaussRenderOutputTest {
	
	Localizer fl;
	GaussRenderOutput gro;
	QueueStore<Localization> localizations; 
	
	@Before
	public void setUp() throws Exception {
		
		String[] arg = {""};
		ImageJ.main(arg);
		
		fl = new FileLocalizer("FileLocalizer.txt");
		localizations = new QueueStore<Localization>();
		gro = new GaussRenderOutput();
		
		fl.setOutput(localizations);
		gro.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(fl).start();
		new Thread(gro).start();
		while (true) {}		
	}

}
