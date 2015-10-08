package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ReadFittedLocalizations;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.plugins.GaussRenderer;

public class GaussRendererTest {

	private Pipeline pipe;
	private ReadFittedLocalizations reader;
	private FastStore localizations;
	private GaussRenderer gauss;

	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");
		
		localizations = new FastStore();
		reader = new ReadFittedLocalizations(new File("/home/ronny/Bilder/fitted.csv"),",");
		reader.setOutput(localizations);
		pipe.add(reader);
		
		gauss = new GaussRenderer(1024, 1024, 0, 127, 0, 127);
		gauss.setInput(localizations);
		pipe.add(gauss);
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,localizations.isEmpty());
	}

}
