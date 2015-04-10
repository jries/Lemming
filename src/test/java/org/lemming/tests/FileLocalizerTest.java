package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import java.io.FileReader;
import java.util.Properties;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.inputs.FileLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

/**
 * Test class for reading localizations from a file. 
 * Prints localizations to the screen.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class FileLocalizerTest {
	
	Properties p;
	FileLocalizer fl;
	PrintToScreen ps;	
	QueueStore<Localization> localizations;

	@Before
	public void setUp() throws Exception {		
		p = new Properties();
		p.load(new FileReader("test.properties"));
		
		ps = new PrintToScreen();
		
		localizations = new QueueStore<Localization>();

		ps.setInput(localizations);
	}

	@Test
	public void testCSV() {
		fl = new FileLocalizer(p.getProperty("samples.dir")+"FileLocalizer.csv");
		fl.setOutput(localizations);

		new Thread(fl).start();
		
		LemMING.pause(1000);
		
		assertEquals(10, localizations.getLength());

		new Thread(ps).start();
		
		LemMING.pause(100);
		
		assertEquals(localizations.getLength(), 0);
	}

	@Test
	public void testTXT() {
		fl = new FileLocalizer(p.getProperty("samples.dir")+"FileLocalizer.txt");
		fl.setOutput(localizations);

		new Thread(fl).start();
		
		LemMING.pause(1000);
		
		assertEquals(10, localizations.getLength());

		new Thread(ps).start();
		
		LemMING.pause(100);
		
		assertEquals(localizations.getLength(), 0);
	}

}
