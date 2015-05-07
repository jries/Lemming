package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.PeakFinder;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.Pipeline;

@SuppressWarnings("rawtypes")
public class PeakFinderTest {

	private Pipeline pipe;
	private FastStore<Frame> frames;
	private IJTiffLoader tif;
	private FastStore<Localization> localizations;
	private PeakFinder peak;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();	
		
		frames = new FastStore<Frame>();
		tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif","frames");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		localizations = new FastStore<Localization>();
		peak = new PeakFinder(700,4,"frames","locs");
		peak.setInput("frames", frames);
		peak.setOutput("locs", localizations);
		pipe.add(peak);
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println(peak.getProcessingTime());		
		assertEquals(true,frames.isEmpty());
	}

}
