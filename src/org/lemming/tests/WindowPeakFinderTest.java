package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.FileReader;
import java.util.Properties;

import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.Store;
import org.lemming.inputs.ScifioLoader;
import org.lemming.outputs.PrintToScreen;
import org.lemming.processors.PeakFinder;
import org.lemming.processors.WindowPeakFinder;
import org.lemming.utils.LemMING;

/**
 * Test class for finding peaks based on a threshold value and inserts the
 * pixel values in a Window around the peak, the frame number and the x,y 
 * coordinates of the localization into a Store. 
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class WindowPeakFinderTest {
	
	ScifioLoader<UnsignedShortType> tif;
	WindowPeakFinder<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> peak;
	PrintToScreen print;

	@Before
	public void setUp() throws Exception {
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
		
		tif = new ScifioLoader<>(p.getProperty("samples.dir")+"eye.tif");
		peak = new WindowPeakFinder<>(200);
		print = new PrintToScreen();
	}

	@Test
	public void test() {
                while (tif.hasMoreOutputs()) {
                        print.process(peak.process(tif.newOutput()));
                }
	}

}
