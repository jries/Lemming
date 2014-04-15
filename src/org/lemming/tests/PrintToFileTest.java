package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
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
		DummyFrameProducer i = new DummyFrameProducer();
		DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d = new DummyLocalizer<>();
		
                while (i.hasMoreOutputs()) {
                        for (Localization localization : d.process(i.newOutput())) {
                                p1.process(localization);
                                p2.process(localization);
                        }
                }
	}
	
}
