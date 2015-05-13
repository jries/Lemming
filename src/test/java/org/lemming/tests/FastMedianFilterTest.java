package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.FastMedianFilter;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.SaveImages;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;

@SuppressWarnings("rawtypes")
public class FastMedianFilterTest {

	private Pipeline pipe;
	private FastStore frames;
	private IJTiffLoader tif;
	private FastMedianFilter fmf;
	private SaveImages saver;
	private FastStore filtered;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();
		
		frames = new FastStore();
		tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif","frames");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		filtered = new FastStore();
		fmf = new FastMedianFilter(50, "frames", "filtered");
		fmf.setInput("frames", frames);
		fmf.setOutput("filtered", filtered);
		pipe.add(fmf);
		
		saver = new SaveImages("/home/ronny/Bilder/out.tif", "filtered");
		saver.setInput("filtered", filtered);
		pipe.add(saver);
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,frames.isEmpty());
	}

}
