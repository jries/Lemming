package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.dummy.QueueStore;
import org.lemming.outputs.PrintToFile;

public class PrintToFileTest {

	PrintToFile p1,p2;
	
	@Before
	public void setUp() throws Exception {
		p1 = new PrintToFile(new File("Test1.csv"));
		p2 = new PrintToFile(new File("Test2.csv"));
	}

	@Test
	public void test() {
		QueueStore<Frame> frames = new QueueStore<>();
		QueueStore<Localization> localizations = new QueueStore<Localization>();
		
		DummyFrameProducer i = new DummyFrameProducer();
		DummyLocalizer d1 = new DummyLocalizer();
		
		i.setOutput(frames);
		d1.setInput(frames);
		d1.setOutput(localizations);
		p1.setInput(localizations);
		p2.setInput(localizations);
		
		new Thread(i).start();
		new Thread(d1).start();
		new Thread(p1).start();
		new Thread(p2).start();
		
		try {
			Thread.sleep(1000);
			
			assertEquals(localizations.getLength(), 0);
			assertEquals(frames.getLength(), 0);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}
