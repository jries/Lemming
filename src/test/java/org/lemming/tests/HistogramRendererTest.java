package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ReadLocalizations;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.HistogramRenderer;

public class HistogramRendererTest {

	private Manager pipe;
	private ReadLocalizations reader;
	private HistogramRenderer histo;

	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		reader = new ReadLocalizations(new File("/Users/ronny/ownCloud/storm/nmsfinder.csv"),",");
		pipe.add(reader);
		
		histo = new HistogramRenderer(1024, 1024, 0, 127, 0, 127);
		pipe.add(histo);
		pipe.linkModules(reader, histo, true);
		
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,pipe.get().values().iterator().next().isEmpty());
	}

}
