package org.lemming.tests;

import java.io.File;
import java.io.IOException;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.DataTable;
import org.lemming.modules.ImageLoader;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.PeakFinder;
import org.lemming.plugins.QuadraticFitter;

import ij.IJ;
import ij.ImagePlus;
import ij.io.FileInfo;
import ij.io.TiffDecoder;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;

public class ExtendableTableTest2 {
	
	private Manager pipe;
	private DataTable dt;
	private ImagePlus loc_im;

	@SuppressWarnings("rawtypes")
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
File file = new File("/Users/ronny/ownCloud/storm/p500ast.tif");
        
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
		
		PeakFinder peak = new PeakFinder(700,6);
		QuadraticFitter fitter = new QuadraticFitter(10);
		dt = new DataTable(); 
		
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(dt);
		
		pipe.linkModules(tif, peak, true);
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,dt);
		
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println(dt.getTable().columnNames().toString());
		System.out.println(dt.getTable().getNumberOfRows());
	}

}
