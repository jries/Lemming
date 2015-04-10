package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.outputs.PrintToFile;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a Store of localizations and then writing the 
 * localizations to a file.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class PrintToFileTest {

	PrintToFile p1,p2;
	
	@Before
	public void setUp() throws Exception {
		p1 = new PrintToFile(new File("PrintToFileTest1.csv"));
		p2 = new PrintToFile(new File("PrintToFileTest2.csv"));
	}

	@Test
	public void test() {
		QueueStore<ImgLib2Frame<UnsignedShortType>> frames = new QueueStore<>();
		QueueStore<Localization> localizations = new QueueStore<Localization>();
		
		DummyFrameProducer i = new DummyFrameProducer();
		DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d1 = new DummyLocalizer<>();
		
		i.setOutput(frames);
		d1.setInput(frames);
		d1.setOutput(localizations);
		p1.setInput(localizations);
		p2.setInput(localizations);
		
		new Thread(i).start();
		new Thread(d1).start();
		new Thread(p1).start();
		new Thread(p2).start();
		
		LemMING.pause(1000);

		assertEquals(localizations.getLength(), 0);
		assertEquals(frames.getLength(), 0);
	}
	
}
