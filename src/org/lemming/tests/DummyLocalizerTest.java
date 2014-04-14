package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.utils.LemMING;

/**
 * Test class for inserting dummy frames into a Store and dummy 
 * localizations into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class DummyLocalizerTest {

	DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d;
	QueueStore<ImgLib2Frame<UnsignedShortType>> frames;
	QueueStore<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {
		d = new DummyLocalizer<>();
		frames = new QueueStore<>();
		localizations = new QueueStore<Localization>();
	}

	@Test
	public void test() {
		DummyFrameProducer i = new DummyFrameProducer();
                DummyLocalizer dummy_localizer = new DummyLocalizer<>();
		
                int localization_count = 0;
                while (i.hasMoreOutputs()) {
                    Array<Localization> localizations =
                        dummy_localizer.process(i.newOutput());
                    ++localization_count;
                }
		assertEquals(localization_count, 200);
	}

}
