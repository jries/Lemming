package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.HistogramRenderer;
import org.lemming.modules.ReadLocalizations;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;

public class HistogramRendererTest {

	private Pipeline pipe;
	private FastStore localizations;
	private ReadLocalizations reader;
	private HistogramRenderer histo;

	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();
		
		localizations = new FastStore();
		reader = new ReadLocalizations(new File("/home/ronny/Bilder/fitted.csv"),",");
		reader.setOutput("locs", localizations);
		pipe.add(reader);
		
		histo = new HistogramRenderer(1024, 1024, 0, 127, 0, 127);
		histo.setInput("locs", localizations);
		pipe.add(histo);	
		
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,localizations.isEmpty());
	}

}
