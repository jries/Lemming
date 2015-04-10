package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.FastStore;
import org.lemming.data.Pipeline;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.HistogramRender;
import org.lemming.processors.StoreSplitter;
import org.lemming.utils.LemMING;

@SuppressWarnings("javadoc")
public class PipelineTest {

	private Pipeline pipe;

	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();
		
	}

	@Test
	public void test() {
		FastStore<Localization> localizations = new FastStore<Localization>();
		RandomLocalizer rl = new RandomLocalizer(50000, 256, 256);
		rl.setOutput(localizations);
		pipe.addSequential(rl);
		
		StoreSplitter<Localization> splitter = new StoreSplitter<Localization>();
		Store<Localization> localizationsCopy1 = new FastStore<Localization>();
		Store<Localization> localizationsCopy2 = new FastStore<Localization>();
		
		splitter.setInput(localizations);
		splitter.addOutput(localizationsCopy1);
		splitter.addOutput(localizationsCopy2);
		
		pipe.add(splitter);
		GaussRenderOutput ren = new GaussRenderOutput(256, 256);
		ren.setInput(localizationsCopy1);
		pipe.add(ren);
		
		HistogramRender hren = new HistogramRender();
		hren.setInput(localizationsCopy2);
		pipe.add(hren);
		pipe.run();
		LemMING.pause(2000);
	}

}
