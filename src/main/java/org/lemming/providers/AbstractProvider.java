package org.lemming.providers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import org.lemming.interfaces.PluginInterface;
import org.scijava.Context;
import org.scijava.InstantiableException;
import org.scijava.log.LogService;
import org.scijava.plugin.PluginInfo;
import org.scijava.plugin.PluginService;

public class AbstractProvider< K extends PluginInterface > {

	private final Class<K> cl;
	private ArrayList<String> keys;
	private ArrayList<String> visibleKeys;
	private ArrayList<String> disabled;
	private HashMap<String, K> implementations;

	AbstractProvider(final Class<K> cl) {
		this.cl = cl;
		registerModules();
	}

	private void registerModules() {
		final Context context = new Context(LogService.class, PluginService.class);
		final LogService log = context.getService(LogService.class);
		final PluginService pluginService = context.getService(PluginService.class);
		final List<PluginInfo<K>> infos = pluginService.getPluginsOfType(cl);

		final Comparator<PluginInfo<K>> priorityComparator = new Comparator<PluginInfo<K>>() {
			@Override
			public int compare(final PluginInfo<K> o1, final PluginInfo<K> o2) {
				return o1.getPriority() > o2.getPriority() ? 1 : o1.getPriority() < o2.getPriority() ? -1 : 0;
			}
		};
		
		Collections.sort(infos, priorityComparator);

		keys = new ArrayList<>(infos.size());
		visibleKeys = new ArrayList<>(infos.size());
		disabled = new ArrayList<>(infos.size());
		implementations = new HashMap<>();

		for (final PluginInfo<K> info : infos) {
			if (!info.isEnabled()) {
				disabled.add(info.getClassName());
				continue;
			}
			try {
				final K implementation = info.createInstance();
				final String key = implementation.getKey();

				implementations.put(key, implementation);
				keys.add(key);
				if (info.isVisible()) 
					visibleKeys.add(key);

			} catch (final InstantiableException e) {
				log.error("Could not instantiate " + info.getClassName(), e);
			}
		}
	}
	
	private List< String > getKeys()
	{
		return new ArrayList<>( keys );
	}

	public List< String > getVisibleKeys()
	{
		return new ArrayList<>( visibleKeys );
	}

	private List< String > getDisabled()
	{
		return new ArrayList<>( disabled );
	}

	public K getFactory( final String key )
	{
		return implementations.get( key );
	}
	
	String echo()
	{
		final StringBuilder str = new StringBuilder();
		str.append("Discovered modules for ").append(cl.getSimpleName()).append(":\n");
		str.append( "  Enabled & visible:" );
		if ( getVisibleKeys().isEmpty() )
		{
			str.append( " none.\n" );
		}
		else
		{
			str.append( '\n' );
			for ( final String key : getVisibleKeys() )
			{
				str.append("  - ").append(key).append("\t-->\t").append(getFactory(key).getName()).append('\n');
			}
		}
		str.append( "  Enabled & not visible:" );
		final List< String > invisibleKeys = getKeys();
		invisibleKeys.removeAll( getVisibleKeys() );
		if (invisibleKeys.isEmpty()) {
			str.append( " none.\n" );
		} else{
			str.append( '\n' );
			for ( final String key : invisibleKeys )
			{
				str.append("  - ").append(key).append("\t-->\t").append(getFactory(key).getName()).append('\n');
			}
		}
		str.append( "  Disabled:" );
		if ( getDisabled().isEmpty() )
		{
			str.append( " none.\n" );
		}
		else
		{
			str.append( '\n' );
			for ( final String cn : getDisabled() )
			{
				str.append("  - ").append(cn).append('\n');
			}
		}
		return str.toString();
	}

}
