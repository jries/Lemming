package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.Fitter;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveFittedLocalizations;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.Manager;
import org.lemming.pipeline.Settings;
import org.lemming.plugins.AstigFitter;
import org.lemming.plugins.PeakFinder;

import ij.IJ;
import ij.ImagePlus;
import ij.io.FileInfo;
import ij.io.TiffDecoder;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;

@SuppressWarnings("rawtypes")
public class ManagerTest2 {

	private Manager pipe;
	private Map<Integer, Store> storeMap;
	private ImagePlus loc_im;
	
	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		
        File file = new File(System.getProperty("user.home")+"/ownCloud/storm/p500ast.tif");
        
		if (file.isDirectory()){
        	FolderOpener fo = new FolderOpener();
        	fo.openAsVirtualStack(true);
        	loc_im = fo.openFolder(file.getAbsolutePath());
        }
        
        if (file.isFile()){
        	File dir = file.getParentFile();
        	TiffDecoder td = new TiffDecoder(dir.getAbsolutePath(), file.getName());
        	FileInfo[] info;
			try {info = td.getTiffInfo();}
    		catch (IOException e) {
    			String msg = e.getMessage();
    			if (msg==null||msg.equals("")) msg = ""+e;
    			IJ.error("TiffDecoder", msg);
    			return;
    		}
    		if (info==null || info.length==0) {
    			IJ.error("Virtual Stack", "This does not appear to be a TIFF stack");
    			return;
    		}
        	FileInfoVirtualStack fivs = new FileInfoVirtualStack(info[0], false);
        	loc_im = new ImagePlus(file.getName(),fivs);
        }
	
	    if (loc_im ==null)
		    throw new Exception("File not found");
		
		ImageLoader tif = new ImageLoader(loc_im);
		//ImageLoader tif = new ImageLoader(new ImagePlus("/Users/ronny/ownCloud/storm/p500ast.tif"));

		PeakFinder peak = new PeakFinder(700,4);
		//QuadraticFitter fitter = new QuadraticFitter(10);
		Fitter fitter = new AstigFitter(7, Settings.readCSV(System.getProperty("user.home")+"/ownCloud/storm/calTest.csv").get("param"));

		UnpackElements unpacker = new UnpackElements();
		SaveFittedLocalizations saver = new SaveFittedLocalizations(new File(System.getProperty("user.home")+"/ownCloud/storm/fitted.csv"));
		SaveLocalizations saver2 = new SaveLocalizations(new File(System.getProperty("user.home")+"/ownCloud/storm/outOrig.csv"));
		
		pipe = new Manager();
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(unpacker);
		pipe.add(saver);
		pipe.add(saver2);
		
		pipe.linkModules(tif, peak, true);
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,saver);
		pipe.linkModules(peak,unpacker);
		pipe.linkModules(unpacker,saver2);
		storeMap = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
	}

}
